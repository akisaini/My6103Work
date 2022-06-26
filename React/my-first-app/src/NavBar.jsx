import React, {Component} from "react";
import { Link } from "react-router-dom";
class NavBar extends Component{
 render(){
    return(<React.Fragment>
<nav class="navbar navbar-expand-lg bg-dark">
  <div class="container-fluid">
    <a class="navbar-brand" href="/#">eCommerce</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarSupportedContent">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        <li class="nav-item">
          <Link className="nav-link" to='/'>Login</Link>
        </li>
        <li class="nav-item">
          <Link className="nav-link" to='/dashboard'>Dashboard</Link>
        </li>
        <li class="nav-item">
          <Link className="nav-link" to='/customers'>Customers</Link>
        </li>
        <li class="nav-item">
          <Link className="nav-link" to='/cart'>Shopping Cart</Link>
        </li>
      </ul>
    </div>
  </div>
</nav>
    </React.Fragment>
    )
 }
}


export default NavBar;