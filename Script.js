document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();  // Prevent page reload

    let fileInput = document.getElementById("fileInput").files[0];
    if (!fileInput) {
        alert("Please select a file before uploading.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);

    let progressBar = document.getElementById("progressBar");
    let progressText = document.getElementById("progressPercentage");
    progressBar.style.width = "0%";
    progressText.innerText = "0%";

    // Hide all pages initially
    document.querySelectorAll("[id^=page]").forEach(page => page.style.display = "none");

    fetch("/upload", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            // Set progress to 100%
            progressBar.style.width = "100%";
            progressText.innerText = "100%";

            // =========================
            // Display Predicted Class
            // =========================
            document.getElementById("predictedClassLabel").innerText =
                "Predicted Class: " + data.predicted_class;
            document.getElementById("classLabel").innerText = data.predicted_class;

            // =========================
            // Display Images
            // =========================
            document.getElementById("originalImageWithLabel").src = "/uploads/" + data.original;
            document.getElementById("gradcamImage").src = "/uploads/" + data.gradcam;
            document.getElementById("limeImage").src = "/uploads/" + data.lime;
            document.getElementById("limeBarChart").src = "/uploads/" + data.lime_bar_chart;

            // =========================
            // Display filenames above images
            // =========================
            document.getElementById("originalFilename").innerText = data.original;
            document.getElementById("gradcamFilename").innerText = data.gradcam;
            document.getElementById("limeFilename").innerText = data.lime;
            document.getElementById("limeBarChartFilename").innerText = data.lime_bar_chart;

            // =========================
            // Set Download Links
            // =========================
            document.getElementById("downloadOriginalWithLabel").href = "/uploads/" + data.original;
            document.getElementById("downloadGradCAM").href = "/uploads/" + data.gradcam;
            document.getElementById("downloadLIME").href = "/uploads/" + data.lime;
            document.getElementById("downloadLIMEBarChart").href = "/uploads/" + data.lime_bar_chart;

            // Show first page
            showPage(1);
        })
        .catch(error => {
            alert("Upload failed! Check the backend logs.");
            console.error("Error:", error);
        });
});

// Function to show a specific page
function showPage(pageNumber) {
    let pages = document.querySelectorAll("[id^=page]");
    pages.forEach(page => page.style.display = "none");
    document.getElementById("page" + pageNumber).style.display = "block";
}
