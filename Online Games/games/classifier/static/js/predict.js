$(document).ready(function() {

   $("#main1").fadeIn(1500,function(){
     $(".start-slide").each(function(){
       var sl_id = "#next_"+$(this).attr("id");
       $(this).slideDown('slow',function(){
         $(sl_id).slideDown('slow');
       });
     });
     $(".hr-line").slideDown('slow');
   });

   $("#predict-btn").click(function(){
     $(".errors").slideUp();
     
     var inputs = [];
     $(".p_input").each(function(){
       inputs.push($(this).val());
     });
     var i;
     var error = [];
     for(i=0;i<inputs.length;i++){
       if (inputs[i] == "" || inputs[i] == null ){
         error.push(i);
       }
     }
     for(i=0;i<error.length;i++){
       var eno = error[i] + 1;
       eid = "#err_"+ eno;
       $(eid).slideDown('slow');
     }
     if(error.length>0){
     }
     else{
       $("#pred").html("P R E D I C T I N G . . .");
       $("#predict-btn").removeClass("hover-btn");
       $("#predict-btn").addClass("loading-btn");
       var ip_str = "";
       for(i=0;i<inputs.length;i++){
         ip_str = ip_str+inputs[i]+";";
       }
       $('#predict_input').prop("value",ip_str);

         $("#submit-btn").click();
     }


   });



});
