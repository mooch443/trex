//var background = new Image();
var date = new Date (); 
var startTime = date.getTime (); 
var objects_order = [], objects_array = {};

function adjust_size(w, h, ctx) {
	var ratio = w / h;
	var max_h = ($(window).height() - $("#menu_bar").height()) * 0.95;
	var max_w = $(window).width() * 0.95;
	
	var sx, sy;
	sx = max_w / w;
	sy = max_h / h;
	
	if(sx < sy) {
		sy = sx;
	} else {
		sx = sy;
	}
	
	$("canvas").width(w*sx).height(h*sy);
	ctx.canvas.width  = w*sx;
	ctx.canvas.height = h*sy;
	
	return {x: sx, y: sy};
}

function toSignedShort(num) {
	return (((((num >>> 8) & 0xFF) << 8) | (num & 0xFF)) << 16) >> 16;
}

function toVec2(num) {
	var y = toSignedShort(num & 0xFFFF);
	var x = toSignedShort((num >>> 16) & 0xFFFF);
	
	return {x: x, y: y};
}

function rshift(num, bits) {
	return num / Math.pow(2,bits);
}

function toVec4(num) {
	var y = num & 0xFFFF;
	var x = rshift(num, 16) & 0xFFFF;
	var height = rshift(num, 32) & 0xFFFF;
	var width = rshift(num, 48) & 0xFFFF;
	
	return {x: x, y: y, w: width, h: height};
}

function toColor(num) {
    num >>>= 0;
    var a = (num & 0xFF) / 255.0,
        b = (num & 0xFF00) >>> 8,
        g = (num & 0xFF0000) >>> 16,
        r = ( (num & 0xFF000000) >>> 24 ) ;
    return "rgba(" + [r, g, b, a].join(",") + ")";
}
function toRGBA(num) {
    num >>>= 0;
    var a = (num & 0xFF) / 255.0,
        b = (num & 0xFF00) >>> 8,
        g = (num & 0xFF0000) >>> 16,
        r = ( (num & 0xFF000000) >>> 24 ) ;
    return {r:b, g:g, b:r, a:a};
}
function get_a(num) {
	return 1.0;
	num >>>= 0;
	return ( (num & 0xFF) ) / 255 ;
}

function toVertices(str, off) {
	var a = String(str).split(',');
	var r = [];
	for(var i in a) {
		var v = toVec2(parseInt(a[i]));
		r.push(v.x+off.x);
		r.push(v.y+off.y);
	}
	r.join(",");
	return r;
}

var types = {1: 'vertices', 2: 'circle', 3: 'rect', 4: 'text', 5: 'image'};
var initial_draw = true;

function update(data) {
	var str = "";
	if(initial_draw)
		str = "/initial";
	initial_draw = false;
	
	$.get("/gui"+str, function(data) {
		if(data.length == 0) {
			setTimeout(update, 100);
			return;
		}
		
		data = JSON.parse(data);
		
		// Get the canvas element using the DOM
	    var canvas = document.getElementById('mycanvas');
	
	    // Make sure we don't execute when canvas isn't supported
	    if (canvas.getContext){
		    var objects = [];
		    var loading_images = 0, found_images = false;
		    
		    var trigger_draw = function() {
			    // use getContext to use the canvas for drawing
		        var ctx = canvas.getContext('2d');
		        var scale = adjust_size(data.w, data.h, ctx);
				
				ctx.fillStyle = "black";
				ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
				
				//ctx.setTransform(1, 0, 0, 1, 0, 0);
				//ctx.scale(scale.x, scale.y);
				//ctx.drawImage(background, 0, 0);
				
				for(var key in objects_array) {
					if(objects_order.indexOf(key) === -1) {
						//console.log("deleting "+key);
						delete objects_array[key];
					}
				}
				
				for(var i=0; i<objects_order.length; i++) {
					var key = objects_order[i];
					var o = objects_array[key];
					
					if(o !== undefined) {
						var matrix = o.matrix;
						
						ctx.setTransform(1, 0, 0, 1, 0, 0);
						ctx.scale(scale.x, scale.y);
						ctx.transform(matrix[0], matrix[1], matrix[2], 
									  matrix[3], matrix[4], matrix[5]);
						o.draw(ctx);
					}
				}
			}
	       
	        for(var i in data.o) {
				var o = data.o[i];
				
				var id = o[0];
				var name = types[id];
				var rect = null;
				var array_index = 1;
				var origin = null;
				var vertices_type = null;
				
				matrix = o[array_index++];
				var pos = {x:0, y:0};
				
				if(name != 'vertices' && name != 'text' && name != 'image') {
					dim = toVec2(parseInt(o[array_index++]));
					rect = {x:0, y:0, w:dim.x, h:dim.y};
				}
				
				if(name == 'vertices')
					vertices_type = o[array_index++];
					
				var obj = null;
					
				switch(name) {
					case 'circle': {
						var clr = toColor(o[array_index++]);
						obj = {
							stroke: clr,
							radius: rect.w*0.5
						};
						
						if(o.length > array_index) {
							obj.fill = toColor(o[array_index++]);
						}
						
						obj.draw = function(ctx) {
							ctx.strokeStyle = this.stroke;
							ctx.beginPath();
							ctx.arc(this.radius, this.radius, this.radius, 0, 2 * Math.PI, false);
							ctx.lineWidth = 1;
							ctx.stroke();
							
							if(this.fill !== undefined) {
								ctx.fillStyle = this.fill;
								ctx.fill();
							}
						};
						
						break;
					}
					case 'vertices': {
						var vertices = toVertices(o[array_index++], pos);
						var clr = toColor(o[array_index++]);
						var thickness = 1;
						
						if(vertices_type == 'S' && o.length < array_index)
							thickness = o[array_index++];
						
						obj = {
							clr: clr,
							thickness: thickness,
							vertices: vertices,
							type: vertices_type
						};
						
						if(vertices_type == 'L') {
							obj.draw = function(ctx) {
								var vertices = this.vertices;
								ctx.strokeStyle = this.clr;
								ctx.fillStyle = this.clr;
								ctx.lineWidth = this.thickness;
								
								for(var i=0; i<vertices.length; i+=4) {
									var a = [vertices[i], vertices[i+1], vertices[i+2], vertices[i+3]];
									ctx.beginPath();
									ctx.moveTo(a[0], a[1]);
									ctx.lineTo(a[2], a[3]);
									ctx.closePath();
									ctx.stroke();
								}
							};
							
						} else {
							obj.draw = function(ctx) {
								var vertices = this.vertices;
								ctx.strokeStyle = this.clr;
								ctx.fillStyle = this.clr;
								ctx.lineWidth = this.thickness;
								
								ctx.beginPath();
								ctx.moveTo(vertices[0], vertices[1]);
								
								for(var i=2; i<vertices.length; i+=2) {
									ctx.lineTo(vertices[i], vertices[i+1]);
								}
								if(this.type == "T")
									ctx.fill();
								else
									ctx.stroke();
							};
						}
						
						break;
					}
					
					case 'rect': {
						obj = {
							fill: toColor(o[array_index++]),
							stroke: toColor(o[array_index++]),
							rect: rect
						};
						
						obj.draw = function(ctx) {
							ctx.fillStyle = this.fill;
							ctx.strokeStyle = this.stroke;
							ctx.fillRect(0, 0, this.rect.w, this.rect.h);
							if(this.stroke !== 'rgba(0, 0, 0, 0)')
								ctx.strokeRect(0, 0 ,this.rect.w, this.rect.h);
						};
						
						break;
					}
					
					case 'text': {
						var text = o[array_index++];
						var fontSize = o[array_index++] * 25;
						var fontFamily = 'MyFont';
						var clr = toColor(o[array_index]);
						
						obj = {
							text: text,
							size: fontSize,
							family: fontFamily,
							clr: clr
						};
						
						obj.draw = function(ctx) {
							ctx.fillStyle = this.clr;
							ctx.font = this.size+"px "+this.family;
							ctx.textBaseline="top"; 
							ctx.fillText(this.text, 0, 0);
						};
						
						break;
					}
					
					case 'image': {
						loading_images++;
						found_images = true;
						
						var image = new Image();
						image.onload = function() {
							loading_images = loading_images - 1;
							if(loading_images == 0) {
								trigger_draw();
							}
						}
						
						var str = o[array_index++];
						var clr = {a: 1};
						
						if(o.length > array_index) {
							clr = toRGBA(parseInt(o[array_index++]));
							console.log(clr);
						}
						
						image.src = 'data:image/png;base64,'+str;
						
						obj = {
							image: image,
							clr: clr
						}
						
						obj.draw = function(ctx) {
							ctx.globalAlpha = this.clr.a;
							ctx.drawImage(this.image, 0, 0);
							ctx.globalAlpha = 1;
						}
						
						break;
					}
				}
				
				if(obj !== null) {
					obj.matrix = matrix;
					objects_array[i] = obj;
				}
			}
			
			objects_order = data.a;
			
			if(!found_images) {
				trigger_draw();
			}
			
	    } else 
	    	console.log("Error");
	    
	    setTimeout(update, 75);
	    
	}).fail(function() {
	    setTimeout(update, 1000);
	});
}

$(document).ready(function() {
	$(document).on("keypress", function(e) {
        if(document.activeElement.tagName == "BODY") {
			if(e.charCode == 8 || e.charCode == 13)
				return;
			
			console.log("char", e.charCode);
	        $.get("/keypress/"+e.charCode, function(data) {
		        data = data == "1" ? true : false;
	        });
        }
    });
    
    $(document).on("keydown", function(e) {
        if(document.activeElement.tagName == "BODY") {
	        $.get("/keycode/"+e.keyCode, function(data) {
		        data = data == "1" ? true : false;
	        });
	    }
    });
    
    $(document).on("mousemove", function(e) {
        if(e.target.tagName == "CANVAS") {
	        var x = (e.clientX - $("canvas").offset().left) / $("canvas").width();
	        var y = (e.clientY - $("canvas").offset().top) / $("canvas").height();
	        
	        if(x >= 0 && x <= 1.0 && y >= 0 && y <= 1.0) {
				var date_now = new Date (); 
				var time_now = date_now.getTime (); 
				var time_diff = time_now - startTime; 
				var seconds_elapsed = time_diff / 1000.0; 
				
				if(seconds_elapsed >= 0.1) {
					startTime = time_now;
					
					$.get("/mousemove/"+x+"/"+y, function(data) {});
				}
	        }
	    }
    });
    
    $("canvas").mousedown(function() {
        $.get("/mousedown");
    }).mouseup(function() {
        $.get("/mouseup");
    });
    
    $("#quit").click(function() {
		$.post("/setting/", {
			"terminate" : "true"
		});
	});
	
	/*background.onload = function() {
		update();
	}
	background.src = "/background.jpg";*/
	
	update();
});