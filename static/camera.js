const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
// const resultInput = document.getElementById('result');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
let stream;
let captureInterval;

// Function to start the camera
startCameraButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Start capturing frames every 5 seconds
        captureInterval = setInterval(() => {
            captureAndPredict();
        }, 5000);
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
});

// Function to stop the camera
stopCameraButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        clearInterval(captureInterval);
    }
});

// Function to capture a frame and predict expression
async function captureAndPredict() {
    if (!stream) return;

    // Draw the current video frame to the canvas
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            // Send the captured frame to the server for prediction
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                resultInput.value = data.result; // Update the result input
            } else {
                console.error('Prediction failed:', response.statusText);
            }
        } catch (error) {
            console.error('Error during prediction:', error);
        }
    }, 'image/jpeg');
}