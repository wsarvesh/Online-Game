$(document).ready(function() {

   $("#main1").fadeIn(800, function(){
     $("#main").slideDown(1000,function(){
       $("#open").click();
     });
   });

   $("#pred").click(function(){
     console.log("HELLO");
     $("#pred-btn").click();
   });

   $("#down").click(function(){
     console.log("DOWNLOAD");
     $("#down-btn").click();
   });


   $(".class-card").each(function(){
     var id = "#"+$(this).attr("id");
     var idnum = $(this).attr("id");
     var cardid = "#instruct"+ idnum;
     var contentid = cardid+"_inside";
     var chartid = "#chartarea"+ idnum;
     var acc_id = "#acc_" + idnum;
     var details = "#details_" + idnum;
     $(id).click(function(){
      if(!$(contentid).is(":visible")){
        var pie = ".pie"+idnum;
        if ($(pie).length == 0){
          var canv = '<canvas class="pie'+idnum+'" id="PieChart'+idnum+'"></canvas>';
          $(chartid).append(canv);
        }
        $(cardid).animate({width:'100%'}, 'slow', function(){
          $(contentid).slideDown("slow", function(){
            // $(cardid).removeClass('instruct-btn');
            $(chartid+"_btn").click();
            countUp(acc_id);
            $(details).slideDown("slow",function(){
              var prf_id = ["#prf1_"+idnum,"#prf2_"+idnum,"#prf3_"+idnum];
              var i;
              for(i=0;i<=3;i++){
                countUp(prf_id[i]);
              }
            });
          });
        });
      }
      else if(!$(contentid).is(":hidden")){
       $(details).slideUp("slow",function(){
         var prf_id = ["#prf1_"+idnum,"#prf2_"+idnum,"#prf3_"+idnum];
         var i;
         for(i=0;i<=3;i++){
           $(prf_id[i]).text("0.0");
         }
         $(contentid).slideUp('slow',function(){
           $(cardid).animate({width:'80%'},'slow',function(){
             // $(cardid).addClass('instruct-btn');
             $(acc_id).text("0.0");
             var pie = ".pie"+idnum;
             if ($(pie).length == 1){
               $(pie).remove();
             }
             if ($(pie).length == 0){
               var canv = '<canvas class="pie'+idnum+'" id="PieChart'+idnum+'"></canvas>';
               $(chartid).append(canv);
             }

           });
         });
       });


      }
     });
   });

   $(".graph-card").each(function(){
     var id = "#"+$(this).attr("id");
     var idnum = $(this).attr("id");
     var cardid = "#graph"+ idnum;
     var contentid = cardid+"_inside";
     var chartid = "#grapharea"+ idnum;
     $(id).click(function(){
      if(!$(contentid).is(":visible")){
        var pie = ".pie"+idnum;
        if ($(pie).length == 0){
          var canv = '<canvas class="pie'+idnum+'" id="Graph'+idnum+'"></canvas>';
          $(chartid).append(canv);
        }
        $(cardid).animate({width:'100%'}, 'slow', function(){
          $(contentid).slideDown("slow", function(){
            // $(cardid).removeClass('instruct-btn');
           $(chartid+"_btn").click();
          });
        });
      }
      else if(!$(contentid).is(":hidden")){

        $(contentid).slideUp('slow',function(){
          $(cardid).animate({width:'80%'},'slow',function(){
            // $(cardid).addClass('instruct-btn');
            var pie = ".pie"+idnum;
            if ($(pie).length == 1){
              $(pie).remove();
            }
            if ($(pie).length == 0){
              var canv = '<canvas class="pie'+idnum+'" id="Graph'+idnum+'"></canvas>';
              $(chartid).append(canv);
            }

          });
        });

      }
     });
   });

   $("#open").click(function() {

     if(!$(".result_inside").is(":visible")){
       $(".class-card").click();
     }
     else{
       var open = 0;
       $(".result_inside").each(function(){
         var id = $(this).attr("id");
         var inst = $(this).attr("id").split("_");
         var idnum = "#" + inst[0][inst[0].length -1];
         if(!$(this).is(":visible")){
           open = open + 1;
           $(idnum).click();
         }
       });
       if (open == 0){
         $(".class-card").each(function(){
           var id = "#instruct"+$(this).attr("id");
           $(id).attr("style","background: #292d45;--notchSize: 30px;--bcolor:#3f4569;width:100%");
           $(id).animate({opacity: 1}, 500, function(){
             $(id).attr("style","--notchSize: 30px;--bcolor:#3f4569;width:100%");
           });
         });
       }
     }
     $("#main1").animate({opacity: 1}, 1200, function(){
       $("#scroll_button2").click();
     });

   });

   $("#open_graph").click(function() {

     if(!$(".graphs_inside").is(":visible")){
       $(".graph-card").click();
     }
     else{
       var open = 0;
       $(".graphs_inside").each(function(){

         var id = $(this).attr("id");
         var inst = $(this).attr("id").split("_");
         var idnum = "#" + inst[0][inst[0].length -2]+inst[0][inst[0].length -1];
         if(!$(this).is(":visible")){
           open = open + 1;
           $(idnum).click();
         }
       });
       if (open == 0){
         $(".graph-card").each(function(){
           var id = "#graph"+$(this).attr("id");
           $(id).attr("style","background: #292d45;--notchSize: 30px;--bcolor:#3f4569;width:100%");
           $(id).animate({opacity: 1}, 500, function(){
             $(id).attr("style","--notchSize: 30px;--bcolor:#3f4569;width:100%");
           });
         });
       }
     }

     $("#main1").animate({opacity: 1}, 1200, function(){
       $("#scroll_button").click();
     });

   });

   $("#scroll_button").click(function() {
     $("html, body").animate({
         scrollTop: $('html, body').get(0).scrollHeight
     }, 2000);
       });

       $("#scroll_button2").click(function() {
         $("html, body").animate({
             scrollTop: 5 * $("#open")[0].scrollHeight
         }, 500);
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
      var n = Math.random();
      var add = n.toFixed(2);
      $this.text(Math.floor(this.countNum) + parseFloat(add));
    },
    complete: function() {
      $this.text(this.countNum);
    }
  });
}
