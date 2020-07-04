$(document).ready(function() {

   $("#main1").fadeIn(800, function(){
     $("#main").slideDown(1000);
   });

   $("#instruct").click(function(){

    if(!$("#instruct_inside").is(":visible")){
      $('#instruct').animate({width:'60%'}, 'slow', function(){
        $("#instruct_inside").slideDown("slow", function(){
          $('#instruct').removeClass('instruct-btn');
        });
      });
    }
    else if(!$("#instruct-inside").is(":hidden")){
      $("#instruct_inside").slideUp('slow',function(){
        $('#instruct').animate({width:'30%'},'slow',function(){
          $('#instruct').addClass('instruct-btn');
        });
      });

    }
   });

   $(".slide-toggle").click(function(){
     $(".box").animate({
       width: "toggle"
     },'slow');
     $(".big-radio").each(function(){
       var id = "#"+$(this).attr("id");
       if($(id).is(':checked')){
         $(id).prop("checked",false);
       }
     });
   });

   $("#demo").click(function(){
     $("#demo").toggleClass("hover-btn");
     $("#demo").toggleClass("hover-btn-fin");
   });

   $("#load").click(function(){
     $("#loading").html("L O A D I N G . . .");
     $("#load").removeClass("hover-btn");
     $("#load").addClass("loading-btn");
     $("#load-btn").click();

   });

});
