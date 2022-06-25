import React, {Component} from 'react'
import NavBar from './NavBar'
import CustomersList from './CustomersList' 
import ShoppingCart from './ShoppingCart'
import Login from './Login'

export default class App extends Component{
 render()
  {
    return  (
        <React.Fragment>
            <NavBar></NavBar>
            <Login></Login>
        </React.Fragment>
    )
  }

} 