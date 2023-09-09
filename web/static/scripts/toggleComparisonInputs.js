function hideK() {
    const kInput = document.getElementById("comparison_inputs");
    kInput.style.display = "none";
}

function displayK() {
    const kInput = document.getElementById("comparison_inputs");
    kInput.style.display = "block";
}

const toggleK = async() => {
    document.getElementById('comparison_radio_group').addEventListener('change', (event) => {
        if( event.target.id === 'precision_recall_f1') {
            displayK();
        }
        else {
            hideK();
        }
    });
}