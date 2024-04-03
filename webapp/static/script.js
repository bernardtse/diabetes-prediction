document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    const refreshButton = document.getElementById('refresh-button');

    // Function to handle form submission and prediction
    async function handlePrediction(event) {
        event.preventDefault();

        // Validate all required fields
        const requiredFields = ['gender', 'age', 'family_diabetes', 'bmi', 'physicallyactive', 'smoking', 'alcohol', 'sleep', 'soundsleep', 'regularmedicine', 'junkfood', 'stress', 'bpLevel', 'pregancies', 'pdiabetes', 'urinationfreq'];
        const missingFields = requiredFields.filter(field => !form[field].value.trim());
        if (missingFields.length > 0) {
            resultDiv.innerHTML = '<p>Please fill in all required fields</p>';
            return;
        }

        // Proceed with prediction if all required fields are filled
        const inputData = {
            gender: form.gender.value,
            age: form.age.value,
            family_diabetes: form.family_diabetes.value,
            bmi: form.bmi.value,
            physicallyactive: form.physicallyactive.value,
            smoking: form.smoking.value,
            alcohol: form.alcohol.value,
            sleep: form.sleep.value,
            soundsleep: form.soundsleep.value,
            regularmedicine: form.regularmedicine.value,
            junkfood: form.junkfood.value,
            stress: form.stress.value,
            bpLevel: form.bpLevel.value,
            pregancies: form.pregancies.value ? parseInt(form.pregancies.value) : null, // Parse as integer
            pdiabetes: form.pdiabetes.value,
            urinationfreq: form.urinationfreq.value
        };

        try {
            const url = '/predict';
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json; charset=utf-8'
                },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Response data:', data);

            const diabetesStatus = data[0].diabetic; // Access correct property
            const probability = data[0].probability.toFixed(2);
            console.log('Diabetes Status:', diabetesStatus);

            // Display the prediction result
            resultDiv.innerHTML = `<h2>Prediction Result</h2>
                                   <p>${diabetesStatus}</p>
                                   <p>Probability: ${probability}</p>`;

        } catch (error) {
            console.error('Error:', error);

            // Display error message
            resultDiv.innerHTML = '<p>An error occurred. Please try again later.</p>';
        }
    }

    // Add event listener to the form submission
    form.addEventListener('submit', handlePrediction);

    // Add event listener to the refresh button
    refreshButton.addEventListener('click', async function () {
        // Clear all form fields
        const formElements = form.querySelectorAll('input, select');
        formElements.forEach(field => {
            field.value = '';
        });

        // Clear the result div
        resultDiv.innerHTML = '';

        // Fetch predictions again
        form.dispatchEvent(new Event('submit'));
    });
});
