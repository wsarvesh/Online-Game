(function(window) {
  function triggerCallback(e, callback) {
    if(!callback || typeof callback !== 'function') {
      return;
    }
    var files;
    if(e.dataTransfer) {
      files = e.dataTransfer.files;
    } else if(e.target) {
      files = e.target.files;
    }

    callback.call(null, files);
  }
  function makeDroppable(ele, callback) {
    var input = document.createElement('input');
    input.setAttribute('type', 'file');
    input.setAttribute('multiple', true);
    input.style.display = 'none';
    input.addEventListener('change', function(e) {
      triggerCallback(e, callback);
    });
    ele.appendChild(input);

    ele.addEventListener('dragover', function(e) {
      e.preventDefault();
      e.stopPropagation();
      ele.classList.add('dragover');
      var disp = document.querySelector('.disp');
      disp.innerHTML = 'Drop File';
    });

    ele.addEventListener('dragleave', function(e) {
      e.preventDefault();
      e.stopPropagation();
      ele.classList.remove('dragover');
      var disp = document.querySelector('.disp');
      disp.innerHTML = 'Drag files here or click to upload';
    });

    ele.addEventListener('drop', function(e) {
      e.preventDefault();
      e.stopPropagation();
      ele.classList.remove('dragover');
      var disp = document.querySelector('.disp');
      disp.innerHTML = 'Drag files here or click to upload';
      triggerCallback(e, callback);
    });

    ele.addEventListener('click', function() {
      input.value = null;
      input.click();
    });
  }
  window.makeDroppable = makeDroppable;
})(this);
(function(window) {
  makeDroppable(window.document.querySelector('.demo-droppable'), function(files) {

    // var output = document.querySelector('.output');
    // output.innerHTML = '';
    if(files.length == 1){
      var fileInput = document.querySelector('.input');
      fileInput.files = files;
        if(files[0].type.indexOf('application/vnd.ms-excel') === 0) {
          // console.log(fileInput.files);
          var disp = document.querySelector('.disp');
          disp.innerHTML = '<p style="font-size:40px">'+files[0].name+'</p>';
        }
        else{
          alert("Barabar file dal");
          var disp = document.querySelector('.disp');
          disp.innerHTML = 'Drag files here or click to upload';
        }
    }
    else{
      alert("1 file dal");
      var disp = document.querySelector('.disp');
      disp.innerHTML = 'Drag files here or click to upload';
    }

  });
})(this);
