// Global variables
let currentIndex = 0;
let currentCategory = "";
let isReal = null; // null, true, or false
let calcificationSeen = null; // null, true, or false
let realismScore = 5; // Default value set to 5


document.addEventListener('DOMContentLoaded', () => {
    // Retrieve last_index from local storage
    const storedIndex = localStorage.getItem('last_index');
    
    if (storedIndex !== null) {
        currentIndex = parseInt(storedIndex, 10);
    }
    
    // Initialize by fetching the first (or next) image
    fetchNextImage();
});

// Fetch the next image from the server
function fetchNextImage() {
    fetch(`/next_image/${currentIndex}`)  // Use currentIndex here
        .then(response => {
            if (response.ok) {
                return response.json();
            } else if (response.status === 404) {
                return response.json().then(data => {
                    if (data.message === "No more images. You can finalize the evaluation.") {
                        finalizeEvaluation(); // Finalize the evaluation and show metrics
                        return null;
                    } else {
                        throw new Error(data.message || "No more images");
                    }
                });
            } else {
                throw new Error("Error fetching image");
            }
        })
        .then(data => {
            if (data) {
                const imageContainer = document.getElementById('image-container');
                const image = document.createElement('img');
                image.src = data.image_url;  // Path to the image
                imageContainer.innerHTML = ''; // Clear previous image
                imageContainer.appendChild(image);

                // Update global variables
                currentIndex++;  // Increment index here
                currentCategory = data.category;

                // Reset button states
                resetButtonStates();

                // Update the last_index in local storage
                localStorage.setItem('last_index', currentIndex);
            }
        })
        .catch(error => {
            console.error('Error fetching image:', error);
            alert('No more images available or an error occurred.');
        });
}

// Finalize the evaluation and redirect to results page
function finalizeEvaluation() {
    const userId = localStorage.getItem('user_id');

    if (!userId) {
        alert('User ID is not available.');
        return;
    }

    fetch('/finalize_evaluation/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_id: userId })
    })
    .then(response => {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return response.json();
        } else {
            return response.text().then(text => {
                console.warn('Received non-JSON response:', text);
                try {
                    return JSON.parse(text);
                } catch (e) {
                    throw new Error('Unable to parse response as JSON');
                }
            });
        }
    })
    .then(data => {
        sessionStorage.setItem('finalMetrics', JSON.stringify(data));
        window.location.href = '/results/?user_id=' + encodeURIComponent(userId);
    })
    .catch(error => {
        console.error('Error finalizing evaluation:', error);
        alert('Failed to finalize evaluation. Please try again later.');
    });
}

// Handle button marking
function markAsFake() {
    isReal = false;
    updateButtonStates();
}

function markAsReal() {
    isReal = true;
    updateButtonStates();
}

function markCalcificationSeen(value) {
    calcificationSeen = value;
    updateCalcificationButtonStates();
}

// Function to set realism score and update button states
function setRealismScore(score) {
    realismScore = score;
    updateRealismButtonStates();
}


function updateButtonStates() {
    document.getElementById('fake-button').classList.toggle('selected', isReal === false);
    document.getElementById('real-button').classList.toggle('selected', isReal === true);
}

function updateCalcificationButtonStates() {
    document.getElementById('calcification-yes-button').classList.toggle('selected', calcificationSeen === true);
    document.getElementById('calcification-no-button').classList.toggle('selected', calcificationSeen === false);
}

// function updateRealismButtonStates() {
//     document.getElementById('unrealistic-button').classList.toggle('selected', realismScore === 1);
//     document.getElementById('realistic-with-details-button').classList.toggle('selected', realismScore === 2);
//     document.getElementById('uncertain-button').classList.toggle('selected', realismScore === 3);
// }

// Function to update the slider value display and label
function updateSliderValue(value) {
    document.getElementById('sliderValue').textContent = value; // Display the current slider value
    realismScore = parseInt(value, 10); // Update the global realismScore variable
    // updateLabel(realismScore); // Update the label based on the new slider value
}

function updateLabel(score) {
    const label = document.getElementById('dynamicLabel');
    
    if (!label) {
        // Create the label if it doesn't exist
        const labelContainer = document.querySelector('.slider-container');
        const newLabel = document.createElement('div');
        newLabel.id = 'dynamicLabel';
        newLabel.style.marginTop = '10px';
        newLabel.style.fontWeight = 'bold';
        labelContainer.appendChild(newLabel);
    }

    let labelText;
    if (score <= 33) {
        labelText = 'Overall Unrealistic';
    } else if (score <= 66) {
        labelText = 'Overall Realistic';
    } else {
        labelText = 'Can\'t Tell';
    }

    label.textContent = labelText;
}


// Initialize the slider value when the page loads or resets
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('realismSlider').value = realismScore; // Initialize slider
    updateSliderValue(realismScore); // Initialize dynamic value display
});

// Handle form submission
function submitEvaluation() {
    // Retrieve user_id from local storage
    const userId = localStorage.getItem('user_id');
    
    if (!userId) {
        alert('User ID is missing. Please start the test first.');
        return;
    }

    if (isReal === null || calcificationSeen === null || realismScore === null) {
        alert('Please select whether the image is real or fake, if calcification is seen, and the realism score.');
        return;
    }

    // Ensure the image is present in the DOM before accessing its src
    const imageElement = document.querySelector('#image-container img');
    if (!imageElement) {
        alert('No image is currently loaded.');
        return;
    }

    fetch('/evaluate/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'user_id': userId,              // Include user_id in the request
            'image_path': imageElement.src,  // Path of the current image
            'category': currentCategory,     // Category of the current image
            'is_real': isReal,               // Whether the image is real or fake
            'realism_score': realismScore,   // Realism score of the image
            'calcification_seen': calcificationSeen, // Whether calcification was seen
            'index': currentIndex            // Include the current image index
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Evaluation submitted:', data);
        checkForMoreImages(); // Check if there are more images to fetch
    })
    .catch(error => {
        console.error('Error submitting evaluation:', error);
        alert('Error submitting evaluation. Please try again.');
    });
}

// Check if there are more images to fetch, or finalize evaluation
function checkForMoreImages() {
    fetch(`/next_image/${currentIndex + 1}`)
        .then(response => {
            if (response.status === 200) {
                // There are more images, fetch the next one
                fetchNextImage(currentIndex + 1);
            } else if (response.status === 404) {
                // No more images, finalize the evaluation
                finalizeEvaluation();
            } else {
                throw new Error('Error checking for more images');
            }
        })
        .catch(error => {
            console.error('Error checking for more images:', error);
            alert('Error checking for more images. Please try again.');
        });
}

// Reset button states and slider when a new image is loaded
function resetButtonStates() {
    isReal = null;
    calcificationSeen = null;
    realismScore = 50; // Default slider position
    updateButtonStates();
    updateCalcificationButtonStates();
    document.getElementById('realismSlider').value = realismScore; // Set slider to default position
    updateSliderValue(realismScore); // Update the dynamic value display and label
}

// Initialize the app by fetching the first image
// fetchNextImage(currentIndex);

// Remove the range input event listener, as it is no longer needed
// document.getElementById('realism_score').removeEventListener('input', ...);
