# Data from the table for generators 1 to 6
generators = [
    {"a": 100, "b": 200, "d": 10, "Pgmin": 0.05, "Pgmax": 1.5},  # Generator 1
    {"a": 120, "b": 150, "d": 10, "Pgmin": 0.05, "Pgmax": 1.5},  # Generator 2
    {"a": 40, "b": 180, "d": 20, "Pgmin": 0.05, "Pgmax": 1.5},   # Generator 3
    {"a": 60, "b": 100, "d": 10, "Pgmin": 0.05, "Pgmax": 1.5},   # Generator 4
    {"a": 40, "b": 180, "d": 20, "Pgmin": 0.05, "Pgmax": 1.5},   # Generator 5
    {"a": 100, "b": 150, "d": 10, "Pgmin": 0.03, "Pgmax": 1.5}   # Generator 6
]

# Initialize lists to store the calculated costs, x values, and p values
costs = []
x_values = []
p_values = []

# Loop to take p (power) inputs and calculate costs and x for each generator
for i, gen in enumerate(generators):
    print(f"\nEnter the value of p (power) for Generator {i+1} (between {gen['Pgmin']} and {gen['Pgmax']}):")
    p = float(input())
    
    # Ensure p is within the generator's allowed range
    if p < gen["Pgmin"] or p > gen["Pgmax"]:
        print(f"Error: Power value out of range for Generator {i+1}. It must be between {gen['Pgmin']} and {gen['Pgmax']}.")
        continue

    # Store the p value
    p_values.append(p)

    # Calculate the cost for this generator
    cost = (gen["a"] * p ** 2) + (gen["b"] * p) + gen["d"]
    costs.append(cost)  # Store the cost in the list

    # Calculate the value of x using the formula x = (p - Pgmin) / (Pgmax - Pgmin)
    x = (p - gen["Pgmin"]) / (gen["Pgmax"] - gen["Pgmin"])
    x_values.append(x)  # Store the x value in the list

    # Print the calculated cost, p, and x for this cycle
    print(f"Cost for Generator {i+1}: {cost}, p = {p}, x = {x}")

# Print each calculated cost, p value, and x value
print("\nThe calculated costs, p values, and x values are:")
for i, (cost, p, x) in enumerate(zip(costs, p_values, x_values), 1):
    print(f"Generator {i}: Cost = {cost}, p = {p}, x = {x}")

# Calculate and print the total cost
total_cost = sum(costs)
print(f"\nThe total cost is: {total_cost}")

# Calculate and print the total of all p values
total_p = sum(p_values)
print(f"The total of all p values is: {total_p}")
