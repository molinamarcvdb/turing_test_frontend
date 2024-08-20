document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded and parsed');
    const userForm = document.getElementById('user-form');

    if (userForm) {
        console.log('User form found, adding event listener');
        userForm.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission
            
            const formData = new FormData(userForm);
            const formObject = Object.fromEntries(formData.entries());

            console.log('Form data:', formObject);

            // Send a request to get a unique identifier and start the test
            const response = await fetch('/start_test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formObject)
            });
            const data = await response.json();
            
            if (data.success) {
                // Store the unique identifier and last index in local storage
                localStorage.setItem('user_id', data.user_id);
                localStorage.setItem('last_index', data.last_index);  // Store last index
                
                // Redirect to the Turing test page
                window.location.href = '/test';
            } else {
                console.error('Failed to start test:', data.error);
            }
        });
    } else {
        console.error('User form not found');
    }
});
