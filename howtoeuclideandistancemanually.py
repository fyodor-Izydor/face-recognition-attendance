import numpy as np

def calculate_euclidean_distance(known_encodings, face_encoding):
    """
    Calculate the Euclidean distance between a face encoding and a list of known encodings.
    
    Args:
        known_encodings (list): List of known face encodings (each encoding is a numpy array)
        face_encoding (numpy.ndarray): Face encoding to compare against known encodings
        
    Returns:
        numpy.ndarray: Array of Euclidean distances between the face encoding and each known encoding
    """
    if len(known_encodings) == 0:
        return np.empty((0,))

    # Convert inputs to numpy arrays if they aren't already
    known_encodings_array = np.array(known_encodings)
    face_encoding_array = np.array(face_encoding)
    
    # Ensure shapes are correct
    if face_encoding_array.ndim == 1:
        face_encoding_array = face_encoding_array.reshape(1, -1)
    
    # Calculate squared differences
    diff = known_encodings_array - face_encoding_array
    squared_diff = np.square(diff)
    
    # Sum along the feature dimension
    sum_squared_diff = np.sum(squared_diff, axis=1)
    
    # Calculate square root to get Euclidean distance
    distances = np.sqrt(sum_squared_diff)
    
    return distances

# Example usage in your main code:
def compare_faces(known_encodings, face_encoding, tolerance=0.40):
    """
    Compare a face encoding against a list of known encodings.
    
    Args:
        known_encodings (list): List of known face encodings
        face_encoding (numpy.ndarray): Face encoding to compare
        tolerance (float): Maximum distance threshold for a match
        
    Returns:
        tuple: (matches, distances) where matches is a boolean array and distances is a float array
    """
    distances = calculate_euclidean_distance(known_encodings, face_encoding)
    matches = distances <= tolerance
    return matches, distances


#To use this in your existing code, you would replace the face_recognition.face_distance() call with our custom implementation. Here's how to modify the relevant section of your code:

# Replace this line:
# faceDis = face_recognition.face_distance(encoded_face_train, encodeFace)

# With these lines:
faceDis = calculate_euclidean_distance(encoded_face_train, encodeFace)
matchIndex = np.argmin(faceDis)

#The manual implementation:
#1. Takes a list of known encodings and a single face encoding to compare against
#2. Converts inputs to numpy arrays for efficient computation
#3. Calculates the Euclidean distance using the formula: sqrt(sum((x1-x2)²))
#4. Returns an array of distances between the input face encoding and each known encoding

#Key features of this implementation:
#- Uses numpy for efficient vector operations
#- Handles both single and batch comparisons
#- Includes shape validation and error handling
#- Provides a companion compare_faces function that mimics the original library's behavior

# The distance calculation follows these steps:
# 1. Subtract each known encoding from the target encoding
# 2. Square the differences
# 3. Sum the squared differences along the feature dimension
# 4. Take the square root to get the final Euclidean distance
