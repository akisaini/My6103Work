import React, {Component} from 'react'
import { BrowserRouter, Routes, Route } from "react-router-dom"
import Todopage from './todopage'
export default class App extends Component{
 render()
  {
    return  (
      <BrowserRouter>
      <Routes>
        <Route path="/" exact element={<Todopage />} />     
      </Routes>
    </BrowserRouter>
    )
  }

} 

{/*
import React, { Component } from "react";
import { Todopage } from "./todopage";
import { BrowserRouter, Routes, Route } from "react-router-dom"
import './App.css'

function App(){

return (

<div className="App">
    Hello Farmer! 
</div>

)

}
 
export default App;




export default class App extends Component {

render() {

    return  (
    <BrowserRouter> 
        <Routes>
            <Route path="/" exact component={<Todopage/>}/>
        </Routes>
    </BrowserRouter>
      )

}


} // Component parenthesis 

*/}
