$(document).ready(function() {

   $("#main1").fadeIn(800, function(){
     $("#main").slideDown(1000,function(){
       $(".start-slide").each(function(){
         $(this).slideDown("slow");
       });
     });
   });

   $(".slide-toggle").click(function(){
     $(".box").animate({
       width: "toggle"
     },'slow');
   });

   $(".end-wrap").each(function(){
     var id = $(this).attr("id");
     var endid = "#end-"+id;
     $(endid).click(function(){
         var attrid = "#attr-div-"+id;
         var attrid_ip = "#attr-"+id;
         $(attrid).fadeOut("slow",function(){
           $(attrid_ip).prop("checked",false);
           $(".attr-wrap").each(function(){
             var attrsid = "#"+$(this).attr("id");
             if(attrsid != attrid){
               $(attrsid).fadeIn("slow");
             }
           });
         });
     });
   });

   $("#opt").click(function(){
     $("#ebox").attr("style","background:#014143;--notchSize:50px;--bcolor:#014143;width:90%;min-height:20vh;");
     $("#abox").attr("style","background:#014143;--notchSize:50px;--bcolor:#014143;width:90%;min-height:20vh;");
     $("#cbox").attr("style","background:#3f4569;--notchSize:60px;--bcolor:#3f4569;width:90%;height:25vh;");
     $("#sbox").attr("style","background:#3f4569;--notchSize:30px;--bcolor:#3f4569;width:90%;");
     $(this).animate({opacity: 1}, 500, function(){
       $("#ebox").attr("style","--notchSize:50px;--bcolor:#014143;width:90%;min-height:20vh;");
       $("#abox").attr("style","--notchSize:50px;--bcolor:#014143;width:90%;min-height:20vh;");
       $("#cbox").attr("style","--notchSize:60px;--bcolor:#3f4569;width:90%;height:25vh;");
       $("#sbox").attr("style","--notchSize:30px;--bcolor:#3f4569;width:90%;");
     });

   });

   $("#attr-all").change(function(){
     if(!$("#attr-all").is(":checked")){
       $(".attrall").each(function(){
         var ip_id = "#"+$(this).attr("id");
         var arr = ip_id.split("-");
         var idnum = arr[arr.length-1];
         var attrid = "#attr-div-"+idnum;
         if($(attrid).is(":visible")){
           $(ip_id).prop("checked",false);
         }
       });
     }
     else{
       $(".attrall").each(function(){
         var ip_id = "#"+$(this).attr("id");
         var arr = ip_id.split("-");
         var idnum = arr[arr.length-1];
         var attrid = "#attr-div-"+idnum;
         if($(attrid).is(":visible")){
           $(ip_id).prop("checked",true);
         }
       });
     }

   });

   $("#classify-all").change(function(){
     if(!$("#classify-all").is(":checked")){
       $(".classify").each(function(){
         var id = "#"+$(this).attr("id");
         $(id).prop("checked",false);
       });
     }
     else{
       $(".classify").each(function(){
         var id = "#"+$(this).attr("id");
         $(id).prop("checked",true);
       });
     }
   });

   $("#start").click(function(){
     $("#end-error").slideUp("slow");
     $("#attr-error").slideUp("slow");
     $("#class-error").slideUp("slow");

     var err = 0;
     if($('#end-select input:checked').length <= 0){
       $("#end-error").slideDown("slow");
     }
     else{
       err = err + 1;
     }
     if($('#attr-select input:checked').length <= 0){
       $("#attr-error").slideDown("slow");
     }
     else{
       err = err + 1;
     }
     if($('#class-select input:checked').length <= 0){
       $("#class-error").slideDown("slow");
     }
     else{
       err = err + 1;
     }
     if(err === 3){
       $("#start-btn").click();
     }

   });

   $("#data").click(function(){
     $("#end-error").slideUp("slow");
     $("#attr-error").slideUp("slow");
     $("#class-error").slideUp("slow");

     var err = 0;
     if($('#end-select input:checked').length <= 0){
       $("#end-error").slideDown("slow");
     }
     else{
       err = err + 1;
     }
     if($('#attr-select input:checked').length <= 0){
       $("#attr-error").slideDown("slow");
     }
     else{
       err = err + 1;
     }
     if(err === 2){
       $("#data-btn").click();
     }

   });



});
