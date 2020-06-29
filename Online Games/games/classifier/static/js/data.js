$(document).ready(function() {

  $("#main1").fadeIn(800, function(){
    $("#main").slideDown(1000,function(){
      $("#open_summary").click()
    });
  });
   
   $("#open_graph").click(function(){
     var show = $("#graphshow").text();
     $("#graphshow").html("X");
     if(show == "0"){
       $(".show-graph").each(function(){
         var gid = "#graph_"+$(this).attr("id");
         $(this).animate({width:'90%'}, 'slow', function(){
           $(gid).each(function(){
             $(this).slideDown("slow",function(){
               var id = $(this).attr("id");
               var canvasdiv = "#div_"+id;
               var btn = "#btn_"+id;
               var canvas_str = '<canvas class="graphic" id="canvas_'+id+'"></canvas>';
               $(canvasdiv).html(canvas_str);
               $(btn).click(); 
               $("#graphshow").html("1");   
             });
           });
         });
       });
       $(".show-img-graph").each(function(){
         var gid = "#graph_img_"+$(this).attr("id");
         $(this).animate({width:'90%'}, 'slow', function(){
             $(gid).slideDown("slow",function(){
               $("#graphshow").html("1");   
             });
         });
       });
       $("#main1").animate({opacity: 1}, 200, function(){
         $("#scroll_button3").click();
       });
     }
     else if (show == "1") {
         $(".graphs").each(function() {
           var idbreak = $(this).attr("id").split('_')
           var idnum = idbreak[1];
           var canv = "#canvas_"+$(this).attr("id");
           var id = "#"+idnum;
           $(this).slideUp("slow",function(){
             $(canv).each(function(){
               $(this).remove();
             });
             $(id).each(function(){
               $(this).animate({width:'70%'}, 'slow');
               $("#graphshow").html("0");
             });
           });
         });
       
     }
   });
   
   $("#open_summary").click(function(){
     var show = $("#summaryshow").text();
     if(show == "0"){
       $(".start-slide").each(function(){
         $(this).slideDown("slow",function(){
           $(".counter").each(function(){
             countUp(this);
           });
         });
         $(".seperator").each(function(){
           $(this).fadeIn();
         });
       });
       $("#summaryshow").html("1");
     }
     else if(show == "1"){
       $(".start-slide").each(function(){
         $(this).slideUp("slow",function(){
           $(".counter").each(function(){
             $(this).html("0");
           })
         });
         $(".seperator").each(function(){
           $(this).fadeOut();
         });
       });
       $("#summaryshow").html("0");
       
     }
     
     $("#main1").animate({opacity: 1}, 600, function(){
       $("#scroll_button2").click();
     });
     
     
   });
   
   $("#open_visual").click(function(){
     $(".visual").slideDown('slow',function(){
       $("#open_graph").click();
     });
   });
   
   $("#scroll_button").click(function() { 
     $("html, body").animate({ 
         scrollTop: $('html, body').get(0).scrollHeight 
     }, 2000);  
       });
       
   $("#scroll_button2").click(function() { 
     $("html, body").animate({ 
         scrollTop: 5 * $("#open_summary")[0].scrollHeight
     }, 800);  
       });
       
       $("#scroll_button3").click(function() { 
         $("html, body").animate({ 
             scrollTop: ($('html, body').get(0).scrollHeight/4 )*3
         }, 2000);  
           });



});

function countUp(id){
  var $this = $(id),
      countTo = $this.attr('data-count');

  $({ countNum: $this.text()}).animate({
    countNum: countTo
  },
  {
    duration: 1200,
    easing:'linear',
    step: function() {
      $this.text(Math.floor(this.countNum));
    },
    complete: function() {
      $this.text(this.countNum);
    }
  });
}