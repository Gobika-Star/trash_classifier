const cameraButton = document.getElementById("cameraButton");
const cameraContainer = document.getElementById("cameraContainer");
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("captureButton");
const capturedImageInput = document.getElementById("captured_image");

let stream;

cameraButton.addEventListener("click", async () => {
  cameraContainer.classList.remove("hidden");
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  }
});

captureButton.addEventListener("click", () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg");
  capturedImageInput.value = imageData;
  document.getElementById("uploadForm").submit();
});
