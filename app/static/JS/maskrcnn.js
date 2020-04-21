$(document).ready(function () {
  var IMAGE_MIME_REGEX = /^image\/(p?jpeg|gif|png)$/i;

  /* IMAGE URL PASTE HANDLER */
  $("#inputurl").bind("keypress", {}, urlkeypress);

  function urlkeypress(e) {
    //BlockInput('block-input-modal', true);
    var code = e.keyCode ? e.keyCode : e.which;
    var $thi = $(this),
      $inp = $thi.val();
    //console.log($inp);
    if (code == 13) {
      //check Enter
      e.preventDefault();
      $("#inputurl").val(null);

      fetch($inp)
        .then(function (response) {
          if (response.ok) {
            return response.blob();
          } else {
            //console.log(response);
            throw new Error("Network response is not ok.");
          }
        })
        .then(function (b) {
          FillPreview($("#imgpreview"), b);
          //BlockInput('block-input-modal', false);
        })
        .catch(function (error) {
          //BlockInput('block-input-modal', false);
          alert("Can't load this image. Please try another.");
          console.log(
            "There was some problems with fetch operation:" + error.message
          );
        });
    }
  }

  /* IMAGE BROWSE HANDLER */
  $("#inputfile").bind("change", function () {
    //BlockInput('block-input-modal', true);
    let imgfile = this.files[0];
    //console.log(imgfile);
    let fileSize = imgfile.size / 1024 / 1024; // this gives in MB
    if (fileSize > 2) {
      $("#inputfile").val(null);
      alert("{{ _('file is too big. images more than 2MB are not allowed') }}");
      return;
    }
    let ext = $("#inputfile").val().split(".").pop().toLowerCase();
    if ($.inArray(ext, ["jpg", "jpeg", "png", "gif"]) == -1) {
      $("#inputfile").val(null);
      alert("{{ _('only jpeg/jpg/png files are allowed!') }}");
      return;
    }
    FillPreview($("#imgpreview"), imgfile);
  });

  /* IMAGE PASTE HANDLER */
  $(document).on("paste", function (e) {
    if (!e.originalEvent.clipboardData || !e.originalEvent.clipboardData.items)
      return;
    let items = e.originalEvent.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (IMAGE_MIME_REGEX.test(items[i].type)) {
        let img = items[i].getAsFile();
        //console.log(img);
        FillPreview($("#imgpreview"), img);
      }
      //showMessage('No image found on your Clipboard!', true);
    }
  });

  function FillPreview(preview, imgblob) {
    let reader = new FileReader();
    reader.onloadend = function () {
      preview.attr("src", reader.result);
    };
    reader.readAsDataURL(imgblob);
  }

  $("#imgpreview").bind("load", function (e) {
    let proxyUrl = "";
    let ImgBlob = e.target.src;
    //console.log(ImgBlob);
    if (ImgBlob.includes("http")) {
      proxyUrl = "https://cors-anywhere.herokuapp.com/";
    }
    //console.log(ImgBlob);
    FillClassPredict(0, true);
    fetch(proxyUrl + ImgBlob)
      .then((i) => i.blob())
      .then(function (b) {
        let urlpost = "/predict";
        //console.log('URLPost: ', urlpost);
        //console.log(b);
        Predict(b, urlpost)
          .then(function (pred) {
            if (!pred["error"]) {
              FillClassPredict(pred);
            } else {
              alert(pred["error"]);
            }
          })
          .catch(function (error) {
            console.log(
              "There has been problem with fetch operation when predicting:" +
                error.message
            );
          });
      });
  });

  /* Brightness of color detecting to make text contrast*/
  function GetContrastTextColor(background_color) {
    let r,
      g,
      b,
      brightness,
      colour = background_color;
    if (colour.match(/^rgb/)) {
      colour = colour.match(/rgba?\(([^)]+)\)/)[1];
      colour = colour.split(/ *, */).map(Number);
      r = colour[0];
      g = colour[1];
      b = colour[2];
    } else if ("#" == colour[0] && 7 == colour.length) {
      r = parseInt(colour.slice(1, 3), 16);
      g = parseInt(colour.slice(3, 5), 16);
      b = parseInt(colour.slice(5, 7), 16);
    } else if ("#" == colour[0] && 4 == colour.length) {
      r = parseInt(colour[1] + colour[1], 16);
      g = parseInt(colour[2] + colour[2], 16);
      b = parseInt(colour[3] + colour[3], 16);
    }
    brightness = (r * 299 + g * 587 + b * 114) / 1000;
    if (brightness < 125) {
      return "white";
    } else {
      return "black";
    }
  }

  async function Predict(imgpost, urlpost) {
    let form_data = new FormData();
    form_data.append("file", imgpost);
    const response = await fetch(urlpost, {
      method: "POST",
      body: form_data,
    });
    return await response.json();
  }

  function FillClassPredict(pred, empty = false) {
    let c = document.getElementById("detection_canvas");
    c.style.opacity = "1.0";
    let ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    $("#img_seg").attr("src", "");
    let elem = $('.time-elem');
    elem.remove();
    if (!empty) {
      let prediction = JSON.parse(pred.data)
      let cont = $(".preview-predict-form");
      let timeelem = '<p class="time-elem" ' +
                'style="position:absolute; left: 0; top: 50px; padding: 0 10px; ' +
                'color: #000; background-color: rgba(255,255,255,0.3);' +
                'font-size: 24px">' + pred['memory'] + '</p>';
      cont.append(timeelem);
      let imgpreview = document.getElementById("imgpreview");
      c.setAttribute("width", imgpreview.width);
      c.setAttribute("height", imgpreview.height);
      for (let i = 0; i < prediction["boxes"].length; i++) {
        let x1 = Math.round(c.width * prediction["boxes"][i][0]);
        let y1 = Math.round(c.height * prediction["boxes"][i][1]);
        let x2 = Math.round(c.width * prediction["boxes"][i][2]);
        let y2 = Math.round(c.height * prediction["boxes"][i][3]);
        DrawRect(
          ctx,
          x1,
          y1,
          x2,
          y2,
          prediction["colors"][i],
          prediction["labels"][i],
          parseInt(c.height * 0.025)
        ); //text height
      }
      if ("masks" in prediction) {
        $("#img_seg").attr("src", prediction["masks"]);
        $("#img_seg").css("opacity", "0.4"); //transparency of mask
      }
    }
  }

  function DrawRect(
    ctx,
    x1,
    y1,
    x2,
    y2,
    color,
    text = "",
    text_size = 10,
    text_pad = 5,
    shadow = 1
  ) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.setLineDash([5, 3]);
    ctx.font = text_size.toString(10) + "px sans-serif";
    let text_width = ctx.measureText(text).width;
    text_width =
      text_width > parseInt((x2 - x1) * 0.8)
        ? parseInt((x2 - x1) * 0.8)
        : text_width;
    ctx.moveTo(x1 + text_pad, y1);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x1 + text_pad + text_width, y1);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.shadowColor = GetContrastTextColor(color);
    ctx.shadowBlur = shadow;
    ctx.fillText(
      text,
      x1 + text_pad,
      y1 + parseInt(text_size * 0.3),
      text_width
    );
  }
});
