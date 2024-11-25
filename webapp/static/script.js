document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    const refreshButton = document.getElementById('refresh-button');

    // Function to handle form submission and prediction
    async function handlePrediction(event) {
        event.preventDefault();

        // Validate all required fields
        const requiredFields = ['Age', 'Gender', 'FamilyDiabetes', 'PhysicallyActive', 'BMI', 'Smoking', 'Alcohol', 'Sleep', 'SoundSleep', 'RegularMedicine', 'JunkFood', 'Stress', 'BPLevel', 'Pregnancies', 'GDiabetes', 'UrinationFreq'];
        const missingFields = requiredFields.filter(field => !form[field].value.trim());
        if (missingFields.length > 0) {
            resultDiv.innerHTML = '<p>Please fill in all required fields</p>';
            return;
        }

        // Proceed with prediction if all required fields are filled
        const inputData = {
            Age: form.Age.value,
            Gender: form.Gender.value,
            FamilyDiabetes: form.FamilyDiabetes.value,
            PhysicallyActive: form.PhysicallyActive.value,
            BMI: form.BMI.value ? Number(form.BMI.value) : null, // Parse as number
            Smoking: form.Smoking.value,
            Alcohol: form.Alcohol.value,
            Sleep: form.Sleep.value ? Number(form.Sleep.value) : null, // Parse as number
            SoundSleep: form.SoundSleep.value ? Number(form.SoundSleep.value) : null, // Parse as number
            RegularMedicine: form.RegularMedicine.value,
            JunkFood: form.JunkFood.value,
            Stress: form.Stress.value,
            BPLevel: form.BPLevel.value,
            Pregnancies: form.Pregnancies.value ? parseInt(form.Pregnancies.value) : null, // Parse as integer
            GDiabetes: form.GDiabetes.value,
            UrinationFreq: form.UrinationFreq.value
        };
    
        console.log("inputData:", inputData); // check inputData
        
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
            resultDiv.innerHTML = `<h2>Assessment Result</h2>
                                   <h5>${diabetesStatus}</h5>
                                   <h5>Probability: ${probability}</h5>`;

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
