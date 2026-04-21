let selectedFile = null;
let selectedFaceIndex = null;
let USE_MOCK_DATA = true;                    // Turn this to false when the backend is ready
const BACKEND_URL = "http://localhost:5000"; // replace with the backend remote container location later

const dropZone = document.getElementById("drop-zone");

dropZone.onclick = () => {
    document.getElementById("fileInput").click();
};

document.getElementById("fileInput").onchange = (e) => {
    selectedFile = e.target.files[0];
    dropZone.innerHTML = `
  <p>${selectedFile.name}</p>
  <p style="opacity:0.6">${selectedFile.type}</p>
`;
};

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    selectedFile = e.dataTransfer.files[0];             // Still researching how this works with images online
    dropZone.innerHTML = `<p>${selectedFile.name}</p>`;
});


// document.getElementById("mockToggle").onchange = (e) => {
//     USE_MOCK_DATA = e.target.checked;
// };

document.getElementById("runBtn").onclick = runAnalysis;

async function runAnalysis() {
    document.getElementById("status").innerText = "Processing...";

    let data;

    if (USE_MOCK_DATA) {
        data = await loadMockData();
    } else {
        data = await callBackend();
    }

    // document.getElementById("status").innerText =
    //     `Detected ${data.faces.length} faces`;

    renderFaces(data.faces);
    renderMedia(data.media);
}

async function loadMockData() {
    const res = await fetch("./mockData.json");
    console.log(res);
    return await res.json();
}

async function callBackend() {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch( BACKEND_URL, {
        method: "POST",
        body: formData
    });

    return await res.json();
}

function renderFaces(faces) {
    const container = document.getElementById("faces-scroll");
    container.innerHTML = "";

    faces.forEach((face, index) => {
        const card = document.createElement("div");
        card.className = "face-card";

        if (index === selectedFaceIndex) {
            card.classList.add("selected");
        }

        card.innerHTML = `
            <img src="${face.image}" style="width:100%; border-radius:8px;" />
            <p>Age: ${face.age}</p>
            <p>${face.actors[0].name}</p>
        `;

        card.onclick = () => {
            selectedFaceIndex = index;
            renderFaces(faces); // need to rerender the faces strip apparently
            renderFaceDetail(face);
        };

        container.appendChild(card);
    });
    
}
function renderFaceDetail(face) {
    const container = document.getElementById("face-detail");

    container.innerHTML = `
        <h2>Face Breakdown</h2>

        <img src="${face.image}" style="width:200px; border-radius:10px;" />

        <h3>Age Prediction</h3>
        <p>${face.age} years (± ${face.mae})</p>

        <h3>Actor Confidence</h3>
        ${face.actors.map(p => `
            <p>
                ${p.name}  -  ${(p.confidence * 100).toFixed(1)}%
            </p>
        `).join("")}
    `;
}

function renderMedia(media) {
    const container = document.getElementById("media-container");
    container.innerHTML = "<h2>Possible Matches</h2>";

    media.forEach(item => {
        const div = document.createElement("div");

        div.innerHTML = `
            <h3>${item.title} (${item.year})</h3>
            <img src="${item.poster}" style="width:120px; border-radius:8px;" />
        `;

        container.appendChild(div);
    });
}
