document.addEventListener("DOMContentLoaded", function () {
  // Mapping datasets to actions
  const datasetActions = {
      "MorphoMNIST": ["thickness", "intensity", "digit"],
      "CelebA_simple": ["smiling", "eyeglasses"],
      "CelebA_complex": ["age", "gender", "beard", "bald"],
      "ADNI": ["apoE", "age", "sex", "brain_volume", "ventricular_volume", "slice_number"]
  };

  // Mapping dataset labels to dataset folder names
  const datasetNames = {
      "MorphoMNIST": "morphomnist",
      "CelebA_simple": "celeba_simple",
      "CelebA_complex": "celeba_complex",
      "ADNI": "adni"
  };

  // Get references to DOM elements
  const datasetSelector = document.getElementById("datasetSelector");
  const actionSelector = document.getElementById("actionSelector");
  const actionSection = document.getElementById("actionSection");
  const imagesSection = document.getElementById("imagesSection");

  const factualImage = document.getElementById("factualImage");
  const vaeImage = document.getElementById("vaeImage");
  const hvaeImage = document.getElementById("hvaeImage");
  const ganImage = document.getElementById("ganImage");

  let currentIndex; // Variable to hold the current index

  // Random index generator
  function getRandomIndex(current) {
      let newIndex;
      do {
          newIndex = Math.floor(Math.random() * 10) + 1; // Adjust the range as needed
      } while (newIndex === current); // Keep generating until it's different from the current index
      return newIndex;
  }

  // Populate action selector based on dataset
  function populateActionSelector(dataset) {
      actionSelector.innerHTML = ""; // Clear previous options
      const actions = datasetActions[dataset];
      actions.forEach(action => {
          const option = document.createElement("option");
          option.value = action;
          option.textContent = action;
          actionSelector.appendChild(option);
      });

      // Select the first action as the default
      if (actions.length > 0) {
          actionSelector.value = actions[0];
      }
  }

  // Load images based on dataset and action
  function loadImages(dataset, action, index) {
      const datasetFolder = datasetNames[dataset];

      // Load the factual image
      factualImage.src = `static/images/${datasetFolder}/factuals/${index}.png`;
      factualImage.alt = `${dataset} factual image ${index}`;

      // Load images for VAE, HVAE, and GAN
      vaeImage.src = `static/images/${datasetFolder}/VAE/${action}/${index}.png`;
      hvaeImage.src = `static/images/${datasetFolder}/HVAE/${action}/${index}.png`;
      ganImage.src = `static/images/${datasetFolder}/GAN/${action}/${index}.png`;

      vaeImage.alt = `${dataset} VAE image for ${action} ${index}`;
      hvaeImage.alt = `${dataset} HVAE image for ${action} ${index}`;
      ganImage.alt = `${dataset} GAN image for ${action} ${index}`;
  }

    // Function to try another sample
    function tryAnotherSample() {
      const selectedDataset = datasetSelector.value;
      const selectedAction = actionSelector.value;
      const newIndex = getRandomIndex(currentIndex); // Get a new random index different from the current index
      loadImages(selectedDataset, selectedAction, newIndex); // Load images with the same action but new index
      currentIndex = newIndex; // Update current index
      return newIndex; // Return the new index
  }

  // Get reference to the "Try Another Sample" button
  const tryAnotherSampleButton = document.getElementById("tryAnotherSampleButton");

  // Event listener for the "Try Another Sample" button
  tryAnotherSampleButton.addEventListener("click", () => {
      const newSampleIndex = tryAnotherSample(); // Call the function and capture the returned index
      console.log("New sample index:", newSampleIndex); // Log the new index
  });

  // Event listener for dataset selection
  datasetSelector.addEventListener("change", () => {
      const selectedDataset = datasetSelector.value;
      if (selectedDataset) {
          populateActionSelector(selectedDataset);
          currentIndex = getRandomIndex(null); // Generate a new random index (null means no current index)
          loadImages(selectedDataset, actionSelector.value, currentIndex); // Load images with the selected action
          actionSection.style.display = "block";
          imagesSection.style.display = "block";
      }
  });

  // Event listener for action selection
  actionSelector.addEventListener("change", () => {
      const selectedDataset = datasetSelector.value;
      const selectedAction = actionSelector.value;
      loadImages(selectedDataset, selectedAction, currentIndex); // Update images using the same index
  });

  // Set initial state with MorphoMNIST and "thickness" action
  currentIndex = getRandomIndex(null); // Generate a random index (null means no current index)
  populateActionSelector("MorphoMNIST");
  loadImages("MorphoMNIST", "thickness", currentIndex);
  actionSection.style.display = "block";
  imagesSection.style.display = "block";
});
