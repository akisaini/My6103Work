import {Component} from "react";
import Product from './Product'

export default class ShoppingCart extends Component {

    constructor(props){
    super(props)
    this.state = {
        products : [],
    }
}

render (){
return (
    <div className='container-fluid'>
        <h4>
        Shopping Cart    
        </h4>    
        <div className='row'>
           {this.state.products.map((prod) => {
            {/* below we are invoking the Product Component. This the Product Component to access parent properties. We can also assign properties to the child Component (button).  */}
            return ( 
            <Product key = {prod.id} product = {prod} onIncrement = {this.handleIncrement} onDecrement = {this.handleDecrement} onDelete = {this.handleDelete}>
            {/*needed for react internally. Primary key not needed now.*/}
                <button className='btn btn-primary'>Buy Now</button>
            </Product>
            
            )
           })}
        </div>
    </div>
)
}
 
// Executes after constructor and render method of current Component. 

componentDidMount = async() => {
// fetch data from data source
var response = await fetch("http://localhost:5000/products", {methods: 'GET'})

var prods = await response.json()
this.setState({products: prods})
}


handleIncrement = (product, maxValue) => {
let allProducts = [...this.state.products]
let index = allProducts.indexOf(product)

if (allProducts[index].quantity < maxValue) {
    allProducts[index].quantity++
    // update the current state of the Products arr.
    this.setState({products: allProducts})
}



}

handleDecrement = (product, minValue) => {
let allProducts = [...this.state.products]
let index = allProducts.indexOf(product)

if (allProducts[index].quantity > minValue) {
    allProducts[index].quantity--

    // update the current state of the Products arr.
    this.setState({products: allProducts})
}
}


handleDelete = (product) => {
    let allProducts = [...this.state.products]
    let index = allProducts.indexOf(product)


    if (window.confirm('Are you sure to delete?')){
        allProducts.splice(index, 1)
        this.setState({products: allProducts})
    }

}

} //Component Parenthesis