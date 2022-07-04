import React, { Component, useState } from "react"; 
import Header from "./header";
import Form from "./form";

export default class Todopage extends Component {

constructor(props){
    super(props);
    this.state = {
        task : '',
    }
}

render(){

    return(<React.Fragment>
        <div className='container'>
            <div className='app-wrapper'>
                <div>
                    <Header/>
                </div>
                <div>
                    <Form></Form>
                </div>
            </div>

        </div>
        
        </React.Fragment>
    )


}



} // Component Parenthesis
