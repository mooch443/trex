$(document).ready(function() {
	$("#info_symbol").mouseenter(function() {
		$("#info_field").stop().fadeIn();
	}).mouseleave(function() {
		$("#info_field").stop().fadeOut();
	});
	
	$.get("/get_settings/", function(data) {
		var obj = $.parseJSON(data);
		console.log("settings set ",obj);
		for(var key in obj) {
			var val = obj[key];
			console.log("setting "+val);
			$("#value_name").append(
				"<option value='"+key+"' type='"+val+"' id='setting_"+key+"'>"
					+key+" : "+escapeHtml(val)
				+"</option>"
			)
		}
		
		$("#value_name").change();
	});
	
	function update_settings() {
		$.get("/info", function(data) {
			var result = (data);
//			result = result.replace(/\r\n\r\n/g, "</p><p>").replace(/\n\n/g, "</p><p>");
//			result = result.replace(/\r\n/g, "<br />").replace(/\n/g, "<br />");
			$("#info_field").html(result);
		});
	}
	
	setInterval(update_settings, 2000);
	
	var tf = [ "true", "false" ];
	var preset_values = {
		"web_quality": [ 10, 25, 50, 75, 100 ]
	};
	
	var preset_triggers = {
		"output_graphs" : function(str) {
			$.get("/output_functions/", function(data) {
				console.log("graphs", data);
				
				var str = "<select id='graph_value' style='width:260px'>";
				for(var i in data.functions) {
					i = data.functions[i];
					str += "<option value='"+i+"'>"+i+"</option>";
				}
				str+="</select><button id='add_graph'>+</button><button id='add_bones'>add bones</button>";
				
				$("#output_graphs").css({"display":"block"}).html(str);
				$("#add_bones").click(function() {
					$("#graph_value option").each(function() {
						if($(this).val().startsWith("bone")) {
							$("#graph_value").val($(this).val());
							$("#add_graph").click();
						}
					});
					
				});
				$("#add_graph").click(function() {
					var value = $("#graph_value").val();
					try {
						var jsonObj = JSON.parse($('#current_value').val());
						var array = [];
						for(var i in jsonObj) {
							console.log(i);
							array.push(jsonObj[i]);
						}
						
						array.push([value, []]);
						console.log(array);
						
						var jsonPretty = JSON.stringify(Object(array));
						$("#value_value").val(jsonPretty);
						$('#current_value').val(JSON.stringify(Object(array), null, '  '));
						
					} catch(e) {
						console.log("Cannot add to invalid json.");
					}
				});
			});
		}
	}
	
	$('#current_value').bind('input propertychange change', function() {
		if(this.value.includes("{") || this.value.includes("[")) {
			try {
				var jsonObj = JSON.parse(this.value);
				var jsonPretty = JSON.stringify(jsonObj);
				$("#value_value").val(jsonPretty);
			} catch(e) {
				$("#value_value").val(this.value);
			}
		} else {
			$("#value_value").val(this.value);
		}
	});
	
	$('#value_value').bind('input propertychange change', function() {
		if(this.value.includes("{") || this.value.includes("[")) {
			try {
				var jsonObj = JSON.parse(this.value);
				var jsonPretty = JSON.stringify(jsonObj, null, '  ');
				$("#current_value").val(jsonPretty);
			} catch(e) {
				$("#current_value").val(this.value);
			}
		} else {
			$("#current_value").val(this.value);
		}
	});
	
	$("#value_name").change(function() {
		var nam = $("#value_name").val();
		var type = $("#value_name option[value='"+nam+"']").attr("type");
		
		$.get("/setting/"+nam, function(data) {
			$("#value_value, #value_send").css({ display: "inline" });
			$("#value_value_select").css({ display: "none" });
			
			var preset = "";
			if(type == "bool") {
				preset = tf;
			}
			
			for(var i in preset_values) {
				if(i == nam) {
					console.log(preset_values[i]);
					preset = preset_values[i];
					break;
				}
			}
			
			if(preset != "") {
				$("#value_value_select").html("").css({ display: "inline" });
				$("#value_value, #value_send").css({ display: "none" });
				
				for(var j in preset) {
					var v = preset[j];
					var sel = "";
					
					if(v == data)
						sel = "selected='selected'";
					$("#value_value_select").append("<option value='"+v+"' "+sel+">"+v+"</option>");
				}
			}
			
			$("#value_value").val(data).change();
			
			for(var i in preset_triggers) {
				if(i == nam) {
					preset_triggers[i](data);
				}
			}
		});
	});
	
	var update_value_fnc = function() {
		var nam = $("#value_name").val();
		var parent = this;
		if($(parent)[0] == $("#value_send")[0])
			parent = $("#value_value");
		var val = $(parent).val();
		
		var obj = {};
		obj[nam] = val;
		console.log("setting "+nam+" to "+val);
		
		$.post("/setting/", obj, function(data) { console.log("Got back: "+data);  });
	};
	
	$("#value_value_select").change(update_value_fnc);
	$("#value_send").click(update_value_fnc);
});