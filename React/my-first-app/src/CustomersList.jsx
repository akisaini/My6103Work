import { Component } from "react";

export default class CustomersList extends Component{

 state = {pageTitle: 'Customers', 
        customersCount: 5,
        customers: [
            {id:1, name: 'Scott', phone: '123-456', address : {city: 'Lon'}, photo: 'https://picsum.photos/id/1010/60'},
            {id:2, name: 'Jomes', phone: '789-101', address : {city: 'MU'}, photo: 'https://picsum.photos/id/1011/60'},
            {id:3, name: 'Allen', phone: null, address : {city: 'NJ'}, photo: 'https://picsum.photos/id/1012/60'},
            {id:4, name: 'John', phone: '151-617', address : {city: 'SF'}, photo: 'https://picsum.photos/id/1013/60'},
            {id:5, name: 'James', phone: null, address : {city: 'NY'}, photo: 'https://picsum.photos/id/1014/60'},
        ],   
    }


    render(){
        return(
        <div>
            <h4>{this.state.pageTitle}
                <span className='badge badge-secondary m-2'>{this.state.customersCount}
                </span>
                <button className="btn btn-info" onClick={this.onRefreshClick}>Refresh</button>
            </h4> 

            <table className='table'>
              <thead>
                <tr>
                    <th>#</th>
                    <th>Photo</th>
                    <th>Customer Name</th>
                    <th>Phone</th>
                    <th>City</th>
                </tr>
                </thead>  
                <tbody>
                    {this.getCustomerRow()}
                </tbody>
            </table> 
        </div>
        ) 
    }

 /* Arrow function */
onRefreshClick = () =>  
 {
    this.setState({
        customersCount:10,
    })
 }



getCustomerRow = () => {
return (this.state.customers.map((cust, index) => {
    return (
        <tr key = {cust.id}>
            <td>{cust.id}</td>
            <td>
                <img src = {cust.photo}></img>
            {/*Adding Css in table data using style class*/}
                <div>
                    <button className='btn btn-sm btn-secondary' onClick ={ () => {
                        this.onChangePictureClick(cust, index)}}>Change Picture</button>
                </div>6
            </td>
            <td>{cust.name}</td>
            {/*Below translates to: if cust.phone equals null then render 'no phone', otherwise (:) show cust.phone value.
            <td>{(cust.phone==null?'No Phone' : cust.phone)}</td>*/}
            <td>{(cust.phone ? cust.phone : (<div className='bg-warning p-2 text-center'>No Phone</div>))}</td>
           {/* Below method works too. It is JSON fetching.*/} 
            <td>{cust['address']['city']}</td>
        </tr>
        )
    }))
  }

onChangePictureClick = (cust, index) =>
{
 //console.log(cust)
 //console.log(index)    
 var custArr = this.state.customers
 custArr[index].photo = 'https://picsum.photos/id/1015/60'

 // update 'customers' array in the state
 this.setState({
    customers:custArr
    })
}



/*   
customerNameStyle = (custName) =>
{
    if (custName.startsWith('S')) return 'green-highlight border-left'
    else if (custName.startsWith('J')) return 'red-highlight border-right'
    else return ''
} 
*/

} /* Component Parenthesis*/