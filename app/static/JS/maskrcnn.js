    $(document).ready(function() {

    $('#imgpreview').bind('load', function (e) {
        let ImgBlob = e.target.src;
        FillClassPredict(0, true);
        fetch('https://cors-anywhere.herokuapp.com/'+ImgBlob, {headers:{"Content-Type":"text/plain;charset=UTF-8"}}).then(i => i.blob()).then(function (b) {
            let urlpost = '/predict';
            //console.log('URLPost: ', urlpost);

            Predict(b, urlpost)
                .then(function (pred) {
                    if (!pred['error']) {
                        //console.log(pred['prediction']);
                        FillClassPredict(pred);
                    } else {
                        alert(pred['error']);
                    }
                }).catch(function (error) {
                    console.log('There has been problem with fetch operation when predicting:' + error.message);
            });
        });
    });

    /* Brightness of color detecting to make text contrast*/
    function GetContrastTextColor(background_color) {
        let r, g, b, brightness,
            colour = background_color;
        if (colour.match(/^rgb/)) {
            colour = colour.match(/rgba?\(([^)]+)\)/)[1];
            colour = colour.split(/ *, */).map(Number);
            r = colour[0];
            g = colour[1];
            b = colour[2];
        } else if ('#' == colour[0] && 7 == colour.length) {
            r = parseInt(colour.slice(1, 3), 16);
            g = parseInt(colour.slice(3, 5), 16);
            b = parseInt(colour.slice(5, 7), 16);
        } else if ('#' == colour[0] && 4 == colour.length) {
            r = parseInt(colour[1] + colour[1], 16);
            g = parseInt(colour[2] + colour[2], 16);
            b = parseInt(colour[3] + colour[3], 16);
        }
        brightness = (r * 299 + g * 587 + b * 114) / 1000;
        if (brightness < 125) {return ("white");}
        else {return ("black");}
    }


    async function Predict(imgpost, urlpost){
        let form_data = new FormData();
        form_data.append('file', imgpost);
        const response = await fetch (urlpost,{
            method: 'POST',
            body: form_data
        });
        return await response.json();
    }



        function FillClassPredict(pred, empty = false) {
            let prediction = pred['prediction'];
            let c = document.getElementById("detection_canvas");
            c.style.opacity = "1.0";
            let ctx = c.getContext("2d");
            ctx.clearRect(0, 0, c.width, c.height);
            $('#img_seg').attr('src', '');
            if (!empty) {
                let imgpreview = document.getElementById("imgpreview");
                let imagesize = prediction['orig_size'];
                c.setAttribute('width', imgpreview.width);
                c.setAttribute('height', imgpreview.height);
                //$('#imgpreview').css("opacity", "0");
                for (let i = 0; i < prediction['boxes'].length; i++) {
                    let x1 = Math.round(c.width * prediction['boxes'][i][0]);
                    let y1 = Math.round(c.height * prediction['boxes'][i][1]);
                    let x2 = Math.round(c.width * prediction['boxes'][i][2]);
                    let y2 = Math.round(c.height * prediction['boxes'][i][3]);
                    DrawRect(ctx, x1, y1, x2, y2,
                        prediction['colors'][i],
                        //'#FFFFFF',
                        prediction['labels'][i],
                        parseInt(c.height * 0.025)); //text height
                }
                //console.log('Masks exist - ', 'masks' in prediction);
                if ('masks' in prediction) {
                    $('#img_seg').attr('src', prediction['masks']);
                    $('#img_seg').css("opacity", "0.4"); //transparency of mask
                } else {
                    //c.style.opacity = "1";
                }
            }
        }

        function DrawRect(ctx, x1, y1, x2, y2, color, text = '', text_size = 10, text_pad = 5, shadow = 1) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.setLineDash([5, 3]);
            ctx.font = text_size.toString(10) + "px sans-serif";
            let text_width = ctx.measureText(text).width;
            //console.log('text: ', text);
            //console.log('text_size: ', text_size);
            //console.log('text_width: ', text_width);
            //console.log('x1,x2,y1,y2: ', x1,x2,y1,y2);
            text_width = (text_width > parseInt((x2 - x1) * 0.8)) ? parseInt((x2 - x1) * 0.8) : text_width;
            //console.log('text_width: ', text_width);
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
            ctx.fillText(text, x1 + text_pad, y1 + parseInt(text_size * 0.3), text_width);
        }

        //$('#imgpreview').trigger('load')
    });







