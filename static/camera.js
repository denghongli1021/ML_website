const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const resultTextarea = document.getElementById("expression-result");

// 啟用攝像頭
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera: ", err);
        resultTextarea.value = "Error accessing camera";
    }
}

// 每秒捕獲一幀圖片，並進行表情偵測
function startDetection() {
    const context = canvas.getContext("2d");

    setInterval(async () => {
        // 設置畫布大小與視頻相匹配
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // 將視頻的一幀畫到畫布上
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // 獲取畫布的數據
        const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));

        // 發送到伺服器進行表情偵測
        const formData = new FormData();
        formData.append("file", imageBlob);

        try {
            const response = await fetch("/predict", { method: "POST", body: formData });
            const data = await response.json();
            if (response.ok) {
                resultTextarea.value = data.result;
            } else {
                resultTextarea.value = "Error: " + data.error;
            }
        } catch (error) {
            console.error("Error during detection:", error);
            resultTextarea.value = "Error during detection";
        }
    }, 1000); // 每秒捕獲一次
}

startCamera().then(() => startDetection());
