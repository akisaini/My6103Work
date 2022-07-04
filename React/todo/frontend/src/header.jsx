import React, {Component} from "react";
import './App.css'

class Header extends Component{
 render(){
    return(<React.Fragment>
        <div className='header'>
            <h1>Todo-List</h1>
        </div>
    </React.Fragment>
    )
 }
}


export default Header;