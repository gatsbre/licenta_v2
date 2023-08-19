function setActiveMenu(activeMenu) {
    const menuOption = document.querySelector(`li.nav-item a.nav-link[href="/${activeMenu}"]`);
    menuOption.classList.add("active");
}