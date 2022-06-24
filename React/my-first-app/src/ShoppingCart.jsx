import {Component} from "react";
import Product from './Product'

export default class ShoppingCart extends Component {
state = {
    products:[
        {id:1, productName:'iPhone', price: 1000, quantity: 0 },
        {id:2, productName:'Camera', price: 3000, quantity: 0 },
        {id:3, productName:'Samsung QLED TV', price: 5000, quantity: 0 },
        {id:4, productName:'iPad Pro', price: 2000, quantity: 0 },
        {id:5, productName:'XBox', price: 750, quantity: 0 },
        {id:6, productName:'Dell Monitor', price: 900, quantity: 0 },
    ]
}

render (){
return (
    <div className='container-fluid'>
        <h4>
        Shopping Cart    
        </h4>    
        <div className='row'>
           {this.state.products.map((prod) => {
            {/* below is returned to Product Component */}
            return ( <Product
            key = {prod.id} // for react internally  (primary key - not needed nowadays)
            product = {prod} />
        
            )
           })}
        </div>
    </div>
)

}
 
} //Component Parenthesis