const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultInput = document.getElementById('result');

// 啟用攝像頭
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream; // 將視頻流綁定到 video 元素
        video.play(); // 確保 video 播放
    } catch (error) {
        console.error('Error accessing camera:', error);
        resultInput.value = 'Camera access denied.';
    }
}

// 將視頻畫面捕捉到 Canvas
function captureFrame() {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg'); // 轉換為 Base64 圖像
}

// 發送圖片到後端進行表情辨識
async function predictExpression() {
    const imageBase64 = captureFrame();
    const formData = new FormData();
    formData.append('file', dataURLtoBlob(imageBase64), 'frame.jpg');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            resultInput.value = data.result; // 顯示表情辨識結果
        } else {
            resultInput.value = `Error: ${data.error}`;
        }
    } catch (error) {
        console.error('Error predicting expression:', error);
        resultInput.value = 'Error predicting expression.';
    }
}

// Base64 轉換為 Blob
function dataURLtoBlob(dataURL) {
    const [header, base64] = dataURL.split(',');
    const binary = atob(base64);
    const array = [];
    for (let i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    return new Blob([new Uint8Array(array)], { type: 'image/jpeg' });
}

// 定期進行表情辨識
function startPredictionLoop() {
    setInterval(() => {
        predictExpression();
    }, 2000); // 每 2 秒捕捉一次畫面並發送
}

// 啟動攝像頭與表情辨識
startCamera();
startPredictionLoop();
