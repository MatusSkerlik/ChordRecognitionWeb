window.onload = function () {

    let loading = document.getElementById('loading');
    let form = document.getElementById('form');
    let method = form.getAttribute('method');
    let action = form.getAttribute('action');

    function startLoading() {
        loading.style.display = 'block';
    }

    function stopLoading() {
        loading.style.display = 'none';
    }

    function sendFile(file) {
        let ajaxData = new FormData(form);
        let ajax = new XMLHttpRequest();

        ajax.open(method, action, true);

        ajax.onload = function (e) {
            console.log(e);
            stopLoading();
            if (ajax.status === 200) {
                location.assign(ajax.responseURL);
            } else {
                document.body.innerHTML = ajax.responseText;
            }
        };

        ajax.onerror = function (e) {
            stopLoading();
            alert("Error " + (this.responseText ? ": " + this.responseText : ", unknown"));
            location.reload();
        };
        ajax.send(ajaxData);
        startLoading();
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


