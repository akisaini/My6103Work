import { Component } from "react";

export default class MainContent extends Component{

 state = {pageTitle: 'Customers', 
        customersCount: 5,
        customers: [
            {id:1, name: 'Scott', phone: '123-456'},
            {id:2, name: 'Jomes', phone: '789-101'},
            {id:3, name: 'Allen', phone: '121-314'},
            {id:4, name: 'John', phone: '151-617'},
            {id:5, name: 'James', phone: '181-920'},
        ],   
    }


    render(){
        return(
        <div>
            <h4 className='border-bottom m-1 p-1'>{this.state.pageTitle}
                <span className='badge badge-secondary m-2'>{this.state.customersCount}
                </span>
                <button className="btn btn-info" onClick={this.onRefreshClick}>Refresh</button>
            </h4>

            <table className='table'>
              <thead>
                <tr>
                    <th>#</th>
                    <th>Customer Name</th>
                    <th>Phone</th>
                </tr>
                </thead>  
                <tbody>
                    {this.state.customers.map((cust) => {
                        return (
                            <tr>
                                <td>{cust.id}</td>
                                <td>{cust.name}</td>
                                <td>{cust.phone}</td>
                            </tr>
                        )
                    }

                    )}
                </tbody>
            </table>
        </div>
        ) 
    }

 onRefreshClick = () =>
 {
    this.setState({
        customersCount:10,
    })
 }
}