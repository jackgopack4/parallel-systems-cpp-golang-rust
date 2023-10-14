import math
def read_double_list_from_file(file_path):
    double_list = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Assuming the numbers are separated by spaces
                numbers = line.split()[1:]
                tmp_list = []
                for number in numbers:
                    # Convert the string to a double and append to the list
                    tmp_list.append(float(number))
                double_list.append(tmp_list)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

    return double_list

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality")

    squared_diff = [(x - y)**2 for x, y in zip(point1, point2)]
    distance = math.sqrt(sum(squared_diff))
    return distance
# Example usage
sample_file = 'sample_output/random-n16384-d24-c16-answer.txt'
sample_array = read_double_list_from_file(sample_file)
result_file = 'output/random-n16384-d24-c16-result.txt'
result_array = read_double_list_from_file(result_file)
# Print the result
print("Sample array:")
for r in result_array:
    print(r)
print("Result array:")
for s in sample_array:
    print(s)

#print("Array of double-length numbers:", sample_array)
#print("result array",result_array)
removed = []
for r in result_array:
    diff = 1000.00
    idx = -1
    for i,s in enumerate(sample_array):
        tmp = euclidean_distance(r,s)
        if tmp < diff:
            diff = tmp
            idx = i
    print("distance between point and centroid: %f"%diff)
    
