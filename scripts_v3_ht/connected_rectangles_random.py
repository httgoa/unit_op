import numpy as np
import random


def generate_random_array_colored(array_size: int = 64, num_colors: int = 16, max_rectangles: int = 10):
    """
    Generate a list of 2D numpy arrays with randomly drawn rectangles.

    Args:
        array_size (int): Size of each array (array_size x array_size).
        num_colors (int): Maximum number of colors (0 to num_colors-1).
        max_rectangles (int): Max number of rectangles to draw on each array.

    """
    
    # Start with a blank array filled with zeros (background is color 0)
    array = np.zeros((array_size, array_size), dtype=np.int32)

    # Draw random rectangles
    for _ in range(random.randint(max_rectangles//2, max_rectangles)):
        # Random rectangle size
        rect_height = random.randint(1, 12)
        rect_width = random.randint(1, 12)

        # Random top-left position, ensure the rectangle fits within the array
        start_x = random.randint(0, array_size - rect_width)
        start_y = random.randint(0, array_size - rect_height)

        # Random color for the rectangle
        color = random.randint(0, num_colors - 1)

        # Fill the area with the color
        array[start_y:start_y + rect_height, start_x:start_x + rect_width] = color

    return array


# Example usage
"""
if __name__ == "__main__":
    # Generate 5 random arrays
    random_arrays = generate_random_arrays(num_arrays=5, array_size=64, num_colors=16, max_rectangles=500)
    
    # Print the first array to check
    print("Example Generated Array:")
    print(random_arrays[0])

    # Visualize the first array using matplotlib (Optional)
    import matplotlib.pyplot as plt
    plt.imshow(random_arrays[0], cmap="tab20", interpolation="nearest")
    plt.title("Random 2D Array with Rectangles")
    plt.axis("off")
    plt.show()
"""
