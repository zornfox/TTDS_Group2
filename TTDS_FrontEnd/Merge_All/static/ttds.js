// change color depends on the fact or fake
var text = document.getElementById("finalresult").textContent;
if(text === "Fact"){
    document.getElementById("finalresult").style.color = "green";
}else{
    document.getElementById("finalresult").style.color = "red";
}
// console.log(text);
