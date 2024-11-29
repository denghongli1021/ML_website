const imageInput = document.getElementById('image-input');
const resultInput = document.getElementById('result');
const imageDisplay = document.getElementById('image-display');
const randomImagesContainer = document.getElementById('random-images-container');
const processedImagesContainer = document.getElementById('processed-images-container');

function displayRandomImages(images) {
    randomImagesContainer.innerHTML = ''; // 清空之前的圖片
    const placedImages = []; // 用於記錄已放置圖片的位置

    images.forEach(imageUrl => {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.classList.add('random-image');

        let topPosition, leftPosition;
        let isValidPosition = false;

        // 嘗試生成符合條件的隨機位置
        while (!isValidPosition) {
            // 隨機生成位置，遠離中心
            do {
                topPosition = Math.random() * 100;
                leftPosition = Math.random() * 100;
            } while (
                topPosition > 30 && topPosition < 70 && // 避免 Y 軸在 30%-70% 範圍內
                leftPosition > 30 && leftPosition < 70 // 避免 X 軸在 30%-70% 範圍內
            );

            // 檢查與已放置圖片的距離
            isValidPosition = placedImages.every(pos => {
                const distance = Math.sqrt(
                    Math.pow(topPosition - pos.top, 2) + Math.pow(leftPosition - pos.left, 2)
                );
                return distance > 15; // 設定距離閾值（此處為 15，視需求調整）
            });
        }

        // 設置圖片位置
        img.style.top = `${topPosition}%`;
        img.style.left = `${leftPosition}%`;
        img.style.transform = `rotate(${Math.random() * 360}deg)`; // 隨機旋轉

        // 添加到已放置圖片的記錄
        placedImages.push({ top: topPosition, left: leftPosition });

        randomImagesContainer.appendChild(img);
    });
}

imageInput.addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        // 显示图片预览
        const reader = new FileReader();
        reader.onload = function (e) {
            imageDisplay.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        };
        reader.readAsDataURL(file);

        try {
            // 发请求到 /analyze 路由
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (response.ok) {
                // 显示表情识别结果
                resultInput.value = `Detected: ${result.label}`;

                // 更新 processedImagesContainer，确保只显示 uploads/face_0.jpg
                processedImagesContainer.innerHTML = ''; // 清空容器内容
                if (result.processed_face) {
                    const img = document.createElement('img');
                    img.src = `/uploads/face_0.jpg?timestamp=${new Date().getTime()}`;
                    img.alt = "Processed Face";
                    img.style.width = "100px";
                    img.style.height = "100px";
                    img.style.margin = "10px";
                    
                    // 将 face_0.jpg 添加到容器中
                    processedImagesContainer.appendChild(img);
                }
            } else {
                resultInput.value = `Error: ${result.error}`;
            }
        } catch (error) {
            console.error('Error:', error);
            resultInput.value = 'Error occurred while analyzing the image.';
        }
    }
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('image-display').innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                resultInput.value = data.result;
                if (data.result === 'happy') {
                    stopRain();
                    showSun();
                } else if (data.result === 'sad') {
                    hideSun();
                    startRain();
                } else if (data.result === 'angry') {
                    hideSun();
                    stopRain();
                    playAngryGif(); // 播放 GIF
                } else {
                    hideSun();
                    stopRain();
                }
                if (data.images && data.images.length > 0) {
                    displayRandomImages(data.images);
                }
            } else {
                resultInput.value = "Error: " + data.error;
            }
        } catch (error) {
            resultInput.value = "Error: " + error.message;
        }
    }
});

function playAngryGif() {
    const angryGifContainer = document.getElementById('angry-gif-container');
    const angryGif = document.getElementById('angry-gif');

    // 顯示 GIF 容器
    angryGifContainer.style.display = 'block';

    // 確保 GIF 從頭播放
    angryGif.src = '/static/angry.gif?' + new Date().getTime();

    // 設定播放時間（假設 3 秒）
    const gifDuration = 3000; // 替換為實際 GIF 的時長
    setTimeout(() => {
        angryGifContainer.style.display = 'none';
    }, gifDuration);
}