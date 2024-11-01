document.addEventListener("DOMContentLoaded", function() {
    const numImages = 10;
    const carousels = document.querySelectorAll(".results-carousel");

    carousels.forEach((carousel, index) => {
      const directory = carousel.getAttribute("data-directory");

      for (let i = 1; i <= numImages; i++) {
        const item = document.createElement("div");
        item.classList.add("item");

        // Construct the image path and set alt/title attributes
        const imgSrc = `${directory}/${i}.png`;
        const imgAlt = `${directory}: Image ${i}`;
        // const imgTitle = `Image ${i} - Carousel ${index + 1}`;

        // Set the HTML content for each image item
        item.innerHTML = `
          <img src="${imgSrc}" alt="${imgAlt}"/>
        `;

        // Append each item to the current carousel
        carousel.appendChild(item);
      }
    });
  });
