import cv2

# Load the haarcascade classifier for frontal faces
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('images/acdc.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE,
)

# Get a count of the faces detected
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around each detected face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Facial Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
