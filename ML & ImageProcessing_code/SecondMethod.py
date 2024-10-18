import cv2
import numpy as np
import matplotlib.pyplot as plt

# load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        exit()
    return image

# convert BGR image to HSV and apply color mask
def apply_color_mask(image, lower_bound, upper_bound):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask

# Function to process the mask with morphological operations
def process_mask(mask):
    # Apply Gaussian Blur to smooth the mask
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Perform erosion and dilation (Morphological operations)
    eroded = cv2.erode(blurred_mask, None, iterations=1)  # Reduce iterations to avoid losing small fish
    dilated = cv2.dilate(eroded, None, iterations=2)  # Dilate to restore structure

    return dilated

# Function to detect contours and use the first fish as reference for pixels-per-mm calculation
def detect_fish_sizes_with_reference(image, mask, known_size_mm):
    # Find contours in the processed mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        print("No contours found.")
        return image, None

    # Use the first fish  as reference
    reference_contour = max(contours, key=cv2.contourArea)  # Use the largest fish as reference

    # Get the bounding box of the reference fish
    x, y, w, h = cv2.boundingRect(reference_contour)

    # Assume the known size (length) of the reference fish is known (in mm)
    reference_fish_length_px = max(w, h)  # Use the largest dimension of the bounding box (length)
    pixels_per_mm = reference_fish_length_px / known_size_mm

    # Draw the contour and size label on the image for the reference fish
    cv2.drawContours(image, [reference_contour], -1, (0, 255, 0), 2)  # Green contour for reference fish
    size_text = f"Reference: {known_size_mm:.2f} mm"
    cv2.putText(image, size_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Process other fish based on the reference pixels_per_mm
    for contour in contours:
        if contour is reference_contour:
            continue  # Skip the reference fish as we've already processed it

        # Calculate the area of the contour in pixels
        area_in_pixels = cv2.contourArea(contour)
        
        # Convert area from pixels to millimeters using the contour area
        area_in_mm = area_in_pixels / (pixels_per_mm ** 2)

        # Skip very small areas (adjusted threshold in mm²)
        if area_in_mm < 5:  # Lower threshold to include smaller fish
            continue

        # Draw the contour and size label for other fish
        x, y, w, h = cv2.boundingRect(contour)
        size_text = f"Size: {area_in_mm:.2f} mm"
        cv2.putText(image, size_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)  # Blue contour for other fish

    return image, pixels_per_mm

# Function to apply the processed mask to the original image
def apply_mask_to_image(image, mask, background_color=(255, 255, 255)):
    # Create a background with the specified color
    background = np.full_like(image, background_color, dtype=np.uint8)
    
    # Use the mask to keep the fish and replace the background with white (or any color)
    result = np.where(mask[:, :, None].astype(bool), image, background)
    return result

# Function to display images using matplotlib
def display_images(original_image, processed_image, processed_title="Processed Image"):
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Processed image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title(processed_title)
    plt.axis('off')

    # Show the result
    plt.show()

# Main function to run the entire pipeline
def main(image_path, known_size_mm):
    # Load the image
    image = load_image(image_path)

    # Define color range for fish segmentation (adjustable)
    lower_bound = np.array([0, 30, 30])  # Adjust lower bound based on the fish color
    upper_bound = np.array([20, 255, 255])  # Adjust upper bound based on the fish color

    # Apply the color mask
    mask = apply_color_mask(image, lower_bound, upper_bound)

    # Process the mask with morphological operations
    processed_mask = process_mask(mask)

    # Apply the processed mask to the original image with a white background (or any color)
    background_color = (255, 255, 255)  # Set background to white
    result = apply_mask_to_image(image, processed_mask, background_color)

    # Detect fish sizes using the first fish as a reference
    result_with_sizes, pixels_per_mm = detect_fish_sizes_with_reference(result, processed_mask, known_size_mm)

    

    # Display the original and processed images
    display_images(image, result_with_sizes, "Fish Segmentation with Size Detection")

# Run the main function
image_path = r'C:\Users\sahar\OneDrive\Desktop\pic\selected\c.jpg'  # Provide the correct image path
known_fish_size_mm = 20  
main(image_path, known_fish_size_mm)
