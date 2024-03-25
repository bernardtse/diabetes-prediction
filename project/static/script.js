document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent the default form submission

        // Get the form data
        const formData = new FormData(form);

        // Convert form data to JSON
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Send the data to the server for prediction
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(prediction => {
            // Display the prediction result
            const resultElement = document.getElementById("prediction-result");
            resultElement.textContent = `Predicted Outcome: ${prediction.outcome}`;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });
});
