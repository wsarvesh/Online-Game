
// var audio = new Audio('aa.mp3');

function nextturn(){

  // clearInterval(x);
  document.getElementById("demo2").innerHTML = "hi";
  // audio.stop();

}


function myFunction() {
var now1 = new Date().getTime();
var countDownDate = new Date(now1).getTime();
countDownDate = countDownDate + 17000;
//document.getElementById("demo1").innerHTML = countDownDate;

// Update the count down every 1 second
var x = setInterval(function() {

  // Get today's date and time
  var now = new Date().getTime();

  // Find the distance between now and the count down date
  var distance = countDownDate - now;
  //document.getElementById("demo1").innerHTML = distance;

  // Time calculations for days, hours, minutes and seconds
  var days = Math.floor(distance / (1000 * 60 * 60 * 24));
  var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
  var seconds = Math.floor((distance % (1000 * 60)) / 1000);

  // Output the result in an element with id="demo"
  document.getElementById("demo").innerHTML = minutes + "m " + seconds + "s ";

  // If the count down is over, write some text
  if (distance < 0) {
    clearInterval(x);
    document.getElementById("demo").innerHTML = "";
    document.getElementById("wrong").click();
    // document.getElementById("demo").innerHTML = "yo";
    // nextturn();
  }
}, 1000);

}
