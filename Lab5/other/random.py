# import random


def generate_points(num_points):
    points = []
    for index in range(num_points):
        x = random.uniform(0, 4)
        y = random.uniform(0, 4)
        mass = random.uniform(0.1, 10.0)  # Positive mass
        x_vel = random.uniform(-1.0, 1.0)
        y_vel = random.uniform(-1.0, 1.0)

        point = f"{index} {x} {y} {mass} {x_vel} {y_vel}"
        points.append(point)

    return points


# Generate 10,000 points
num_points = 10000
points_data = generate_points(num_points)

# Save to a file or use as needed
with open("generated_points.txt", "w") as file:
    for point_data in points_data:
        file.write(point_data + "\n")
