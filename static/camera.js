const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const flipCameraButton = document.getElementById('flip-camera');
let stream;
let currentFacingMode = 'user'; // 初始為前置攝像頭

// Function to start the camera
startCameraButton.addEventListener('click', async () => {
    await startCamera(currentFacingMode);
});

// Function to stop the camera
stopCameraButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
});

// Function to flip the camera
flipCameraButton.addEventListener('click', async () => {
    if (stream) {
        // 停止當前的攝像頭流
        stream.getTracks().forEach(track => track.stop());
    }
    // 切換攝像頭方向
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    await startCamera(currentFacingMode);
});

// Function to start the camera with a specified facing mode
async function startCamera(facingMode) {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode }
        });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}
