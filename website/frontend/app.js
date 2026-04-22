let selectedFile = null;
let selectedFaceIndex = null;

let USE_MOCK_DATA = true;
const BACKEND_URL = "http://localhost:5000";

const elements = {
    dropZone: document.getElementById("drop-zone"),
    runBtn: document.getElementById("runBtn"),
    mockToggle: document.getElementById("mockToggle"),
    fileInput: document.getElementById("fileInput"),
    status: document.getElementById("status"),
    mediaContainer: document.getElementById("media-container"),
    facesWrapper: document.getElementById("faces-container-wrapper")
};



elements.dropZone.onclick = () => elements.fileInput.click();

elements.fileInput.onchange = (e) => handleFileSelect(e.target.files[0]);

elements.dropZone.ondragover = (e) => e.preventDefault();

elements.dropZone.ondrop = async (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    
    if (files.length > 0) {
        handleFileSelect(files[0]);
    } else {
        const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
        if (url) tryFetchWebImage(url);
    }
};

async function tryFetchWebImage(url) {
    elements.dropZone.innerHTML = `<p>Fetching web image...</p>`;
    try {
        const res = await fetch(url, { cache: 'no-cache' });
        const blob = await res.blob();
        handleFileSelect(new File([blob], "web-image.jpg", { type: blob.type }));
    } catch (err) {
        elements.dropZone.innerHTML = `<p style="color:red">Security Block: Download image manually.</p>`;
    }
}

function handleFileSelect(file) {
    if (!file) return;
    selectedFile = file;
    renderDropZonePreview(file);
    updateButtonState();

    // Update the status text to guide the user
    elements.status.innerText = "Now click Run Analysis or click the box again to select another file.";
    
    // Optional: Add a little color to make it pop
    elements.status.style.color = "#4dabf7"; 
}

function renderDropZonePreview(file) {
    const isVideo = file.type.startsWith('video/');
    const fileUrl = URL.createObjectURL(file);
    
    elements.dropZone.innerHTML = "";
    
    const media = document.createElement(isVideo ? 'video' : 'img');
    media.src = fileUrl;
    media.className = "preview-media"; // Ensure CSS has pointer-events: none
    if (isVideo) {
        media.muted = true;
        media.onloadeddata = () => media.currentTime = 1;
    }

    const info = document.createElement('div');
    info.style.cssText = "position: relative; z-index: 2; pointer-events: none;";
    info.innerHTML = `<strong>${file.name}</strong><br><span style="opacity:0.7">Click to change</span>`;

    elements.dropZone.append(media, info);
}


function updateButtonState() {
    elements.runBtn.disabled = !(selectedFile || elements.mockToggle.checked);
}

elements.mockToggle.onchange = () => {
    USE_MOCK_DATA = elements.mockToggle.checked;
    updateButtonState();
};

elements.runBtn.onclick = async () => {
    elements.status.innerText = "Processing...";
    
    try {
        const data = USE_MOCK_DATA ? await (await fetch("./mockData.json")).json() : await callBackend();
        
        elements.status.innerText = `Detected ${data.faces.length} faces`;
        elements.facesWrapper.style.display = "block";
        elements.mediaContainer.style.display = "block";
        
        renderFaces(data.faces);
        renderMedia(data.media);
    } catch (err) {
        elements.status.innerText = "Error loading data.";
    }
};

async function callBackend() {
    const formData = new FormData();
    formData.append("file", selectedFile);
    const res = await fetch(BACKEND_URL, { method: "POST", body: formData });
    return res.json();
}

function renderFaces(faces) {
    const container = document.getElementById("faces-scroll");
    container.innerHTML = "";

    faces.forEach((face, index) => {
        const card = document.createElement("div");
        card.className = `face-card ${index === selectedFaceIndex ? 'selected' : ''}`;
        card.innerHTML = `
            <img src="${face.image.includes('MOCK') ? 'test.png' : face.image}" style="width:100%; border-radius:8px;" />
            <p>Age: ${face.age}</p>
            <p><strong>${face.actors[0].name}</strong></p>
        `;
        card.onclick = () => {
            selectedFaceIndex = index;
            renderFaces(faces);
            renderFaceDetail(face);
        };
        container.appendChild(card);
    });
}

function renderFaceDetail(face) {
    const container = document.getElementById("face-detail");
    container.style.display = "block";
    container.innerHTML = `
        <h2>Face Breakdown</h2>
        <img src="${face.image.includes('MOCK') ? 'test.png' : face.image}" style="width:200px; border-radius:10px;" />
        <h3>Age Prediction: ${face.age} (± ${face.mae})</h3>
        ${face.actors.map(p => `<p>${p.name} - ${(p.confidence * 100).toFixed(1)}%</p>`).join("")}
    `;
}

function renderMedia(media) {
    elements.mediaContainer.innerHTML = "<h2>Possible Matches</h2>";
    const grid = document.createElement("div");
    grid.className = "media-grid";

    media.forEach(item => {
        const card = document.createElement("a");
        card.className = "movie-card";
        card.href = `https://www.imdb.com/title/tt${item.imdbId}/`;
        card.target = "_blank";
        card.innerHTML = `
            <img src="${item.poster.includes('MOCK') ? 'test.png' : item.poster}" />
            <div class="movie-info"><h3>${item.title}</h3><p>${item.year}</p></div>
        `;
        grid.appendChild(card);
    });
    elements.mediaContainer.appendChild(grid);
}

updateButtonState();