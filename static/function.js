let countryy;
let clist =[];
let cropdata;
let croplist = [];
fetch('http://127.0.0.1:8000/apiendpoint').then((res)=>res.json()).then((d)=>{
  console.log(d)
  console.log(typeof d)
  countryy = Object.keys(d)
  cropdata = Object.values(d)

}).then(()=>
{
  
  let t1;
  let t2;
  for(let i in countryy)
  {
    
    clist.push(countryy[i])
  }
  for(let j in cropdata)
  {
    croplist.push(cropdata[j])

  }

  console.log(clist)
  console.log(croplist)
  let cn1 = document.getElementById('selectcountry');
  
  let crp1 = document.getElementById('selectcrop');
  clist.forEach(function addCountry(c){
    let option = document.createElement("option");
    option.text = c;
    option.value = c;
    cn1.append(option);
  });
  cn1.onchange = function (){
    crp1.innerHTML = "<option></option>";
    console.log('selected val')
    console.log(document.getElementById('selectcountry').value)
    let getv = document.getElementById('selectcountry').value
    for(let c in clist)
    {
      if(clist[c] == getv)
      {
        console.log(getv)
        addToCrop(c)
      }
    }
    console.log(getv)
    
  }
  function addToCrop(k)
  {
        
    croplist[k].forEach(function(it){
      let option = document.createElement("option");
    option.text = it;
    option.value = it;
    crp1.append(option);
    });
  }

})