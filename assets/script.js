const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const ws = new WebSocket(`ws://${window.location.host}/ws`);

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Error accessing webcam:", err));

const labelsContainer = document.getElementById('labels-container');
const gestureImg = document.getElementById('gesture-img');
const fpsVal = document.getElementById('fps-val');
const qualitySlider = document.getElementById('quality-slider');
const qualityVal = document.getElementById('quality-val');
const landmarksCheckbox = document.getElementById('landmarks-checkbox');

qualitySlider.oninput = () => { qualityVal.innerText = qualitySlider.value; };

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Display image
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0);
    img.src = 'data:image/jpeg;base64,' + data.image;
    
    // Update labels
    labelsContainer.innerHTML = data.labels.map(l => `<div class="label-item">${l}</div>`).join('');

    // Update FPS
    if (data.fps) {
        fpsVal.innerText = data.fps;
    }

    // Show/Hide matched gesture image
    if (data.matched_gesture) {
        gestureImg.src = `/assets/images/gestures/${data.matched_gesture.toLowerCase()}.png`;
        gestureImg.style.display = 'block';
    } else {
        gestureImg.style.display = 'none';
        gestureImg.src = '';
    }
};

function sendFrame() {
    if (ws.readyState === WebSocket.OPEN) {
        // Ensure canvas matches internal dimensions for processing
        if (canvas.width !== 640) canvas.width = 640;
        if (canvas.height !== 480) canvas.height = 480;
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const quality = parseFloat(qualitySlider.value);
        const data = canvas.toDataURL('image/jpeg', quality).split(',')[1];
        ws.send(json.stringify({
            image: data,
            show_landmarks: landmarksCheckbox.checked
        }));
    }
    requestAnimationFrame(sendFrame);
}
video.onplay = () => requestAnimationFrame(sendFrame);
