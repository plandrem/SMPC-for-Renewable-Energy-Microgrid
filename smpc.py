import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import trange

class RenewablePowerSMPC:
    """
    SMPC controller for renewable power allocation in a microgrid.
    """
    
    def __init__(self, horizon=24, num_scenarios=10, time_step_minutes=60):
        """Initialize the SMPC controller."""
        self.horizon = horizon
        self.num_scenarios = num_scenarios
        self.time_step = time_step_minutes / 60.0  # Convert to hours
        
        # System parameters
        self.battery_capacity = 500.0  # kWh
        self.battery_max_charge_rate = 100.0  # kW
        self.battery_max_discharge_rate = 100.0  # kW
        self.battery_efficiency = 0.9  # 90% round-trip efficiency
        
        # Cost parameters
        self.datacenter_unmet_penalty = 10.0  # $/kWh
        self.ev_unmet_penalty = 2.0  # $/kWh
    
    def generate_solar_scenarios(self, base_forecast, uncertainty_profile):
        """Generate solar power scenarios based on forecast uncertainty."""
        scenarios = []
        for _ in range(self.num_scenarios):
            # Generate correlated noise
            rho = 0.7  # Temporal correlation
            noise = np.zeros(self.horizon)
            noise[0] = np.random.normal(0, 1)
            
            for t in range(1, self.horizon):
                noise[t] = rho * noise[t-1] + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
            
            # Apply time-of-day dependent uncertainty
            scenario = []
            for t in range(self.horizon):
                # Apply noise scaled by uncertainty
                uncertainty = uncertainty_profile[t]
                perturbed = base_forecast[t] * (1 + uncertainty * noise[t])
                # Ensure non-negative power
                scenario.append(max(0, perturbed))
            
            scenarios.append(scenario)
        
        return scenarios
    
    def solve(self, current_time, solar_forecast, dc_demand_forecast, 
              ev_demand_forecast, battery_soc, uncertainty_profile):
        """
        Solve the SMPC optimization problem.
        
        Args:
            current_time: Current time
            solar_forecast: Base solar power forecast
            dc_demand_forecast: Datacenter demand forecast
            ev_demand_forecast: EV charging demand forecast
            battery_soc: Current battery state of charge (0-1)
            uncertainty_profile: Forecast uncertainty by hour
            
        Returns:
            Dictionary with optimal decisions for the current time step
        """
        # Generate scenarios
        solar_scenarios = self.generate_solar_scenarios(solar_forecast, uncertainty_profile)
        scenario_probs = [1.0 / self.num_scenarios] * self.num_scenarios
        
        # Current battery energy
        initial_battery = battery_soc * self.battery_capacity
        
        # Variables
        dc_power = {}
        ev_power = {}
        battery_charge = {}
        battery_discharge = {}
        battery_energy = {}
        dc_unmet = {}
        ev_unmet = {}
        curtailment = {}
        
        for s in range(self.num_scenarios):
            for t in range(self.horizon):
                dc_power[s, t] = cp.Variable(nonneg=True)
                ev_power[s, t] = cp.Variable(nonneg=True)
                battery_charge[s, t] = cp.Variable(nonneg=True)
                battery_discharge[s, t] = cp.Variable(nonneg=True)
                battery_energy[s, t] = cp.Variable(nonneg=True)
                dc_unmet[s, t] = cp.Variable(nonneg=True)
                ev_unmet[s, t] = cp.Variable(nonneg=True)
                curtailment[s, t] = cp.Variable(nonneg=True)
        
        # Constraints
        constraints = []
        
        # Non-anticipativity constraints (first-stage decisions must be the same)
        for s in range(1, self.num_scenarios):
            constraints.append(dc_power[s, 0] == dc_power[0, 0])
            constraints.append(ev_power[s, 0] == ev_power[0, 0])
            constraints.append(battery_charge[s, 0] == battery_charge[0, 0])
            constraints.append(battery_discharge[s, 0] == battery_discharge[0, 0])
        
        # Power balance and other constraints
        for s in range(self.num_scenarios):
            for t in range(self.horizon):
                # Power balance
                constraints.append(
                    solar_scenarios[s][t] + battery_discharge[s, t] ==
                    dc_power[s, t] + ev_power[s, t] + battery_charge[s, t] + curtailment[s, t]
                )
                
                # Battery limits
                constraints.append(battery_charge[s, t] <= self.battery_max_charge_rate)
                constraints.append(battery_discharge[s, t] <= self.battery_max_discharge_rate)
                
                # Cannot charge and discharge simultaneously
                # (This is a non-convex constraint, we'd need binary variables in practice)
                
                # Battery dynamics
                if t == 0:
                    constraints.append(
                        battery_energy[s, t] == initial_battery + 
                        battery_charge[s, t] * self.battery_efficiency * self.time_step - 
                        battery_discharge[s, t] * self.time_step / self.battery_efficiency
                    )
                else:
                    constraints.append(
                        battery_energy[s, t] == battery_energy[s, t-1] + 
                        battery_charge[s, t] * self.battery_efficiency * self.time_step - 
                        battery_discharge[s, t] * self.time_step / self.battery_efficiency
                    )
                
                # Battery capacity constraints
                constraints.append(battery_energy[s, t] <= self.battery_capacity)
                constraints.append(battery_energy[s, t] >= 0.1 * self.battery_capacity)  # Min SOC 10%
                
                # Load constraints
                constraints.append(dc_power[s, t] + dc_unmet[s, t] >= dc_demand_forecast[t])
                constraints.append(ev_power[s, t] + ev_unmet[s, t] >= ev_demand_forecast[t])
        
        # Objective function
        objective_terms = []
        for s in range(self.num_scenarios):
            for t in range(self.horizon):
                # Cost components
                penalty_cost = (dc_unmet[s, t] * self.datacenter_unmet_penalty +
                               ev_unmet[s, t] * self.ev_unmet_penalty) * self.time_step
                
                # Add to objective with scenario probability
                objective_terms.append(scenario_probs[s] * penalty_cost)
        
        # Create and solve the problem
        objective = cp.Minimize(sum(objective_terms))
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status != "optimal":
                print(f"Warning: Problem status is {problem.status}")
                return None
            
            # Extract first-stage decisions
            return {
                'datacenter_power': dc_power[0, 0].value,
                'ev_power': ev_power[0, 0].value,
                'battery_charge': battery_charge[0, 0].value,
                'battery_discharge': battery_discharge[0, 0].value,
                'expected_cost': problem.value,
                'battery_trajectory': [battery_energy[0, t].value for t in range(self.horizon)]
            }
            
        except Exception as e:
            print(f"Error solving optimization: {e}")
            return None

# Example usage with realistic data
def run_simulation():
    """Run a simulation of the microgrid over a day."""
    # Create controller
    controller = RenewablePowerSMPC(horizon=24, num_scenarios=10)
    
    # Time parameters
    start_time = datetime(2023, 7, 15, 0, 0)  # Summer day
    
    # Create realistic solar profile (summer day)
    hours = np.arange(24)
    solar_capacity = 400  # kW
    solar_profile = solar_capacity * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    
    # Forecast uncertainty increases with forecast horizon
    base_uncertainty = 0.1  # 10% uncertainty
    uncertainty_profile = [base_uncertainty * (1 + 0.1 * t) for t in range(24)]
    
    # Datacenter has relatively constant load
    dc_base_load = 150  # kW
    dc_profile = dc_base_load + 20 * np.sin(np.pi * hours / 12)
    
    # EV charging peaks in morning and evening
    ev_capacity = 100  # kW
    morning_peak = np.exp(-0.5 * ((hours - 8) / 2) ** 2) * ev_capacity
    evening_peak = np.exp(-0.5 * ((hours - 18) / 2) ** 2) * ev_capacity
    ev_profile = morning_peak + evening_peak
    
    # Initial conditions
    battery_soc = 0.5  # 50% charged
    
    # Run simulation
    results = []
    actual_battery = battery_soc * controller.battery_capacity
    
    print("Running simulation over 24 hours:")
    for hour in trange(24):
        current_time = start_time + timedelta(hours=hour)
        
        # Forecast from current time onwards
        solar_forecast = solar_profile[hour:].tolist() + solar_profile[:hour].tolist()
        dc_forecast = dc_profile[hour:].tolist() + dc_profile[:hour].tolist()
        ev_forecast = ev_profile[hour:].tolist() + ev_profile[:hour].tolist()
        
        # Get uncertainty profile
        uncertainty = uncertainty_profile[:]
        
        # Solve SMPC
        decision = controller.solve(current_time, solar_forecast, dc_forecast, 
                                    ev_forecast, battery_soc, uncertainty)
        
        if decision is None:
            print(f"Failed to find solution at hour {hour}")
            continue
        
        # Simulate actual conditions (using base forecast for simplicity)
        actual_solar = solar_profile[hour]
        actual_dc = dc_profile[hour]
        actual_ev = ev_profile[hour]
        
        # Implement the decisions
        dc_power = min(decision['datacenter_power'], actual_dc)
        ev_power = min(decision['ev_power'], actual_ev)
        battery_charge = decision['battery_charge']
        battery_discharge = decision['battery_discharge']
        
        # Update battery state
        actual_battery += (battery_charge * controller.battery_efficiency - 
                          battery_discharge / controller.battery_efficiency) * controller.time_step
        actual_battery = min(max(actual_battery, 0), controller.battery_capacity)
        battery_soc = actual_battery / controller.battery_capacity
        
        # Calculate power balance
        power_balance = (actual_solar + battery_discharge - 
                         dc_power - ev_power - battery_charge)
        
        # Record results
        results.append({
            'hour': hour,
            'time': current_time,
            'solar': actual_solar,
            'datacenter_demand': actual_dc,
            'ev_demand': actual_ev,
            'datacenter_power': dc_power,
            'ev_power': ev_power,
            'battery_charge': battery_charge,
            'battery_discharge': battery_discharge,
            'battery_soc': battery_soc,
            'power_balance': power_balance,
            'expected_cost': decision['expected_cost']
        })
    
    return pd.DataFrame(results)

# Run simulation and plot results
results_df = run_simulation()

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

# Power sources and loads
axs[0].plot(results_df['hour'], results_df['solar'], color='C1', label='Solar Generation')
axs[0].plot(results_df['hour'], results_df['battery_discharge'], color='C2', label='Battery Discharge')
axs[0].plot(results_df['hour'], -results_df['datacenter_power'], color='C0', label='Datacenter Load')
axs[0].plot(results_df['hour'], -results_df['ev_power'], color='C3', label='EV Charging Load')
axs[0].set_ylabel('Power (kW)')
axs[0].set_title('Power Sources and Loads')
axs[0].grid(True, alpha=0.3)
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Demands vs. allocations
axs[1].plot(results_df['hour'], results_df['datacenter_demand'], color='C0', linestyle='--', label='Datacenter Demand')
axs[1].plot(results_df['hour'], results_df['datacenter_power'], color='C0', linestyle='-', label='Datacenter Allocation')
axs[1].plot(results_df['hour'], results_df['ev_demand'], color='C3', linestyle='--', label='EV Demand')
axs[1].plot(results_df['hour'], results_df['ev_power'], color='C3', linestyle='-', label='EV Allocation')
axs[1].set_ylabel('Power (kW)')
axs[1].set_title('Demands vs. Allocations')
axs[1].grid(True, alpha=0.3)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Battery state of charge
axs[2].plot(results_df['hour'], results_df['battery_soc'] * 100, color='C2')
axs[2].set_xlabel('Hour of Day')
axs[2].set_ylabel('Battery SoC (%)')
axs[2].set_title('Battery State of Charge')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(right=0.6)  # Adjust the right margin to make room for the legends
plt.show()