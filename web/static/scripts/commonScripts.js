function setActiveMenu(activeMenu) {
  const menuOption = document.querySelector(
    `li.nav-item a.nav-link[href="/${activeMenu}"]`
  );
  menuOption.classList.add("active");
}

const isFormValid = async () => {
  const form = document.getElementById("evaluationForm");
  const formElements = form.elements;
  const evaluateButton = document.getElementById("evaluate_button");
  let valid = true;
  let checkedBox = false;

  for (let i = 0; i < formElements.length; i++) {
    const element = formElements[i];

    if (element.name === "model" && element.checked) {
      checkedBox = true;
    }

    if (!element.checkValidity()) {
      valid = false;
      break;
    }

    if (element.value === "") {
      valid = false;
      break;
    }
  }

  if (valid && checkedBox) {
    evaluateButton.disabled = false;
  } else {
    evaluateButton.disabled = true;
  }
};

const selectAll = async () => {
  document.getElementById("selectAll").addEventListener("change", (event) => {
    const checkboxes = document.querySelectorAll(
      'input[type="checkbox"][name="model"]'
    );
    for (let i = 0; i < checkboxes.length; i++) {
      checkboxes[i].checked = event.target.checked;
    }
  });
};

const evaluateButton = async () => {
  document
    .getElementById("evaluate_button")
    .addEventListener("click", async (event) => {
      event.preventDefault();
      await plotModels();
    });
};
