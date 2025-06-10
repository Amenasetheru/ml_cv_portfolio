document.addEventListener('DOMContentLoaded', () => {
    // --- Smooth Scrolling Logic for Navigation Links ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            // Check if the link is meant for the project_detail page (which should not smooth scroll on the same page)
            if (this.getAttribute('href').startsWith('#') && !this.classList.contains('project-card-link-button')) { // Updated class name
                e.preventDefault();

                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);

                if (targetElement) {
                    // Calculate offset to account for fixed header
                    const headerOffset = document.querySelector('.main-header').offsetHeight;
                    const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
                    const offsetPosition = elementPosition - headerOffset - 20; // -20 for a little extra padding

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: "smooth"
                    });
                }
            }
        });
    });

    // --- Handle Flash Messages Redirection and Scroll ---
    // If a flash message is present and an anchor is in the URL, scroll to it
    const urlParams = new URLSearchParams(window.location.search);
    const anchor = urlParams.get('_anchor'); // This will capture Flask's _anchor
    if (anchor) {
        const targetElement = document.getElementById(anchor);
        if (targetElement) {
            const headerOffset = document.querySelector('.main-header').offsetHeight;
            const elementPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
            const offsetPosition = elementPosition - headerOffset - 20;

            window.scrollTo({
                top: offsetPosition,
                behavior: "smooth"
            });
            // Clear the anchor from URL after scrolling to prevent re-scrolling on refresh
            // if (history.replaceState) {
            //     const cleanUrl = window.location.origin + window.location.pathname;
            //     history.replaceState(null, '', cleanUrl);
            // }
        }
    }
});

