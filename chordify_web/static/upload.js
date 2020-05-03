window.onload = function () {

    let loading = document.getElementById('loading')
    let form = document.getElementById('form');
    let method = form.getAttribute('method');
    let action = form.getAttribute('action');

    function sendFile(file) {
        let ajaxData = new FormData(form);
        let ajax = new XMLHttpRequest();

        ajax.open(method, action, true);

        ajax.onload = function (e) {
            console.log(e);
            loading.style.display = 'none';
            alert("Success");
            location.reload();
        };

        ajax.onerror = function (e) {
            console.log(e);
            loading.style.display = 'none';
            alert("Error " + (this.responseText ? ": " + this.responseText : ", unknown"));
            location.reload();
        };
        ajax.send(ajaxData);
        loading.style.display = 'block';
    }

    let input = document.getElementById('file');
    let click = document.getElementById('click');

    input.addEventListener('change', function (e) {
        sendFile(e.target.files[0])

    });

    click.addEventListener('click', function (e) {
        input.click()
    });

};


