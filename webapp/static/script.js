document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');

    form.addEventListener('submit', async function (event) {
        event.preventDefault();

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
            pregnancies: form.pregnancies.value,
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

            const diabetesStatus = data.diabetic; // Access correct property
            const probability = data.probability.toFixed(2);
            console.log('Diabetes Status:', diabetesStatus);

            // Display the prediction result
            resultDiv.innerHTML = `<h2>Prediction Result</h2>
                                   <p>Diabetes Status: ${diabetesStatus}</p>
                                   <p>Probability: ${probability}</p>`;
        } catch (error) {
            console.error('Error:', error);

            // Display error message
            resultDiv.innerHTML = '<p>An error occurred. Please try again later.</p>';
        }
    });
});
